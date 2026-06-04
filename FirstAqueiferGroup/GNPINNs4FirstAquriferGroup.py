# -*- coding: utf-8 -*-
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
import xlwt
import tensorflow as tf

# 基础配置与GPU设置
tf.keras.backend.clear_session()
gc.collect()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("GPU 配置失败：", e)

import sciann as sn
from sciann import Variable, Functional, SciModel
import pandas as pd
import xlrd
from mgwr.gwr import GWR
from mgtwr.model import GTWR
from mgwr.sel_bw import Sel_BW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# matplotlib 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)
sn.set_random_seed(42)

# --- 工具函数 ---

def interpolate_h0_kriging(interpolate_points, sites_coords, initial_water_levels):
    try:
        from pykrige.ok import OrdinaryKriging
        H0_values = []
        for i, point in enumerate(interpolate_points):
            is_duplicate = False
            for j, site in enumerate(sites_coords):
                if np.allclose(point, site, rtol=1e-5, atol=1e-5):
                    H0_values.append(initial_water_levels[j])
                    is_duplicate = True
                    break
            if not is_duplicate:
                H0_values.append(None)
        
        need_interpolation = [i for i, val in enumerate(H0_values) if val is None]
        if need_interpolation:
            if len(sites_coords) > 3:
                OK = OrdinaryKriging(
                    sites_coords[:, 0], sites_coords[:, 1], initial_water_levels,
                    variogram_model='spherical', verbose=False, enable_plotting=False
                )
                points_to_interpolate = interpolate_points[need_interpolation]
                z_interp, _ = OK.execute('points', points_to_interpolate[:, 0], points_to_interpolate[:, 1])
                for idx, interp_idx in enumerate(need_interpolation):
                    H0_values[interp_idx] = z_interp[idx]
            else:
                avg_val = np.mean(initial_water_levels)
                for idx in need_interpolation:
                    H0_values[idx] = avg_val
        return np.array(H0_values)
    except Exception as e:
        return np.mean(initial_water_levels) * np.ones(len(interpolate_points))

def read_excel_to_array(file_path):
    workbook = xlrd.open_workbook(file_path)
    sheet = workbook.sheet_by_index(0)
    data = []
    header_cells = sheet.row_values(0)
    has_string_header = any(isinstance(cell, str) and not str(cell).replace('.','',1).isdigit() for cell in header_cells)
    start_row = 1 if has_string_header else 0
    for row_idx in range(start_row, sheet.nrows):
        row_vals = sheet.row_values(row_idx)
        numeric_row = [float(val) if isinstance(val, (int, float)) else np.nan for val in row_vals]
        data.append(numeric_row)
    arr = np.array(data, dtype=np.float64)
    
    if '水位值' in file_path:
        for i in range(arr.shape[0]):
            s = pd.Series(arr[i, :])
            arr[i, :] = s.interpolate(limit_direction='both').values
    else:
        df = pd.DataFrame(arr)
        arr = df.fillna(df.mean()).values
    return arr

# --- 用户路径 ---
BASE_DIR = r"E:\研究生\PINN-Dissertation-4\AAAAAAAAAAAAGWR-PINNs18-245\数据集\潜水"
COORD_FILE = os.path.join(BASE_DIR, r"训练数据\xls文件\潮白河2018-20245坐标.xls")
EXTRACT_FILE = os.path.join(BASE_DIR, r"训练数据\xls文件\潮白河2018-20245开采量.xls")
RAIN_FILE = os.path.join(BASE_DIR, r"训练数据\xls文件\潮白河2018-20245降雨量.xls")
HEAD_FILE = os.path.join(BASE_DIR, r"训练数据\xls文件\潮白河2018-20245水位值.xls")

col_x = 6; col_y = 7; col_S = -2; col_K = -1

def prepare_common():
    coord_array = read_excel_to_array(COORD_FILE)
    extract_array = read_excel_to_array(EXTRACT_FILE)
    rain_array = read_excel_to_array(RAIN_FILE)
    head_array = read_excel_to_array(HEAD_FILE)
    x_all = coord_array[:, col_x]
    y_all = coord_array[:, col_y]
    S_all = coord_array[:, col_S]
    K_all = coord_array[:, col_K]
    nt = min(extract_array.shape[1], rain_array.shape[1], head_array.shape[1])
    return dict(x_all=x_all, y_all=y_all, S_all=S_all, K_all=K_all,
                extract_all=extract_array[:, :nt], rain_all=rain_array[:, :nt], 
                head_all=head_array[:, :nt], nt=nt)

# --- 核心逻辑 ---

def residual_correction_gtwr_pinn():
    print('正在运行增强版 GTWR-PINNs (保留原输出格式)...')
    data = prepare_common()
    x_all, y_all, nt = data['x_all'], data['y_all'], data['nt']
    rain_all, extract_all, head_all = data['rain_all'], data['extract_all'], data['head_all']
    K_all, S_all = data['K_all'], data['S_all']
    
    # 1. 构建数据集
    X_list, H_list, well_indices_list = [], [], []
    for ti in range(nt):
        t_val = float(ti)
        theta = 2.0 * np.pi * (ti % 12) / 12.0
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        for i in range(len(x_all)):
            X_list.append([x_all[i], y_all[i], t_val, sin_t, cos_t, 
                          rain_all[i,ti], extract_all[i,ti], K_all[i], S_all[i]])
            H_list.append(head_all[i,ti])
            well_indices_list.append(i)
            
    X, H, well_indices = np.array(X_list), np.array(H_list).reshape(-1,1), np.array(well_indices_list)

    # 数据分割 (按井分割)
    all_wells = np.unique(well_indices)
    train_wells, test_wells = train_test_split(all_wells, test_size=0.3, random_state=42)
    train_mask, test_mask = np.isin(well_indices, train_wells), np.isin(well_indices, test_wells)
    
    # 归一化
    X_min, X_max = X[train_mask].min(axis=0), X[train_mask].max(axis=0)
    X_range = X_max - X_min + 1e-8
    H_min, H_max = H[train_mask].min(axis=0), H[train_mask].max(axis=0)
    H_range = H_max - H_min + 1e-8
    
    # 保存归一化参数 (确保预测时一致)
    np.savez(os.path.join(BASE_DIR, 'scalers_and_meta_fe_full.npz'), 
             X_min=X_min, X_max=X_max, X_range=X_range,
             H_min=H_min, H_max=H_max, H_range=H_range)
    
    Xtr = (X[train_mask] - X_min) / X_range
    Htr = (H[train_mask] - H_min) / H_range

    # 2. PINN 模型定义
    x = Variable('x'); y = Variable('y'); tv = Variable('t')
    tsin = Variable('tsin'); tcos = Variable('tcos')
    r = Variable('rain'); e = Variable('extract')
    kvar = Variable('k'); svar = Variable('s'); h0_var = Variable('h0')
    
    # 网络优化：8层 x 128神经元，tanh激活，L2正则化
    phys = Functional('nn_phys', [x,y,tv,tsin,tcos,r,e,kvar,svar], 
                      8*[128], 'tanh', kernel_regularizer=tf.keras.regularizers.l2(1e-6))
    
    # 初值演化方程优化：确保 t=0 时严格符合初值，随时间平滑演化
    H_trial_expr = h0_var + (1.0 - sn.math.exp(-tv)) * phys
    H_orig = H_trial_expr * H_range + H_min
    
    # PDE Loss 构造
    h_safe = sn.math.abs(H_orig) + 0.1
    H_t, H_x, H_y = sn.diff(H_orig, tv), sn.diff(H_orig, x), sn.diff(H_orig, y)
    K_Hx, K_Hy = kvar * h_safe * H_x, kvar * h_safe * H_y
    
    recharge = (r * X_range[5] + X_min[5])
    pumping = (e * X_range[6] + X_min[6])
    
    # PDE Residual (引入0.01缩放因子平衡量纲)
    pde_res = (sn.diff(K_Hx, x) + sn.diff(K_Hy, y) + recharge + pumping - svar * H_t) * 0.01
    
    model = SciModel([x,y,tv,tsin,tcos,r,e,kvar,svar, h0_var], [H_trial_expr, pde_res], 
                     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    # H0 准备
    coords_sites = np.vstack([x_all, y_all]).T
    H0_train_orig = interpolate_h0_kriging(X[train_mask][:, 0:2], coords_sites, head_all[:, 0])
    H0_train_s = (H0_train_orig.reshape(-1, 1) - H_min) / H_range

    train_inputs = [Xtr[:, i:i+1] for i in range(Xtr.shape[1])] + [H0_train_s]
    train_targets = [Htr, np.zeros_like(Htr)]
    
    # 训练
    model.train(train_inputs, train_targets, epochs=200, batch_size=1024,
                adaptive_weights={'method': 'NTK', 'freq': 100},
                callbacks=[ReduceLROnPlateau(factor=0.5, patience=150, min_lr=1e-6, verbose=1),
                           EarlyStopping(patience=400, restore_best_weights=True)])
    
    # 【新增】关键步骤：保存训练好的权重文件
    # 这将生成 'baseline_pinn_fixed.h5'，用于预测脚本加载
    weights_path = os.path.join(BASE_DIR, 'baseline_pinn_fixed.h5')
    model.save_weights(weights_path)
    print(f"==========================================")
    print(f"模型权重已成功保存至: {weights_path}")
    print(f"==========================================")

    # 3. 预测与残差修正
    H0_all_orig = interpolate_h0_kriging(X[:, 0:2], coords_sites, head_all[:, 0])
    H0_all_s = (H0_all_orig.reshape(-1,1) - H_min) / H_range
    X_all_s = (X - X_min) / X_range
    inputs_all = [X_all_s[:, i:i+1] for i in range(X_all_s.shape[1])] + [H0_all_s]
    
    H_pred = (model.predict(inputs_all)[0] * H_range + H_min).flatten()
    residuals = H.flatten() - H_pred
    
    # 按井计算平均残差并使用GWR进行空间插值修正
    site_res = np.array([np.mean(residuals[well_indices == i]) for i in range(len(x_all))]).reshape(-1, 1)
    
    try:
        # 特征：截距 + 局部导水系数K
        X_gwr = K_all.reshape(-1, 1)
        sel = Sel_BW(coords_sites, site_res, X_gwr)
        bw = sel.search()
        gwr_res = GWR(coords_sites, site_res, X_gwr, bw).fit().predy.flatten()
    except:
        from scipy.interpolate import Rbf
        rbf = Rbf(coords_sites[:,0], coords_sites[:,1], site_res.flatten(), function='multiquadric', smooth=0.1)
        gwr_res = rbf(coords_sites[:,0], coords_sites[:,1])

    # 最终修正 (引入0.9平滑系数)
    H_corrected = H_pred + np.repeat(gwr_res, nt) * 0.9

    # --- 调用原始输出函数 (格式完全保留) ---
    output_dir = BASE_DIR
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    print("\n=== Baseline PINN 井位结果 ===")
    export_well_results(H, H_pred.reshape(-1,1), well_indices, x_all, y_all, train_wells, test_wells, 
                       output_dir, model_name="Baseline_PINN")
    
    print("\n=== 修正后模型 井位结果 ===")
    export_well_results(H, H_corrected.reshape(-1,1), well_indices, x_all, y_all, train_wells, test_wells, 
                       output_dir, model_name="GTWR_PINN_Corrected")

# --- 原始输出函数模块 (禁止修改格式) ---

def calculate_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    mae = np.mean(np.abs(y_pred - y_true))
    r2 = 1 - np.sum((y_true - y_pred)**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-8)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def calculate_well_metrics(y_true, y_pred, well_indices, x_coords, y_coords, well_names=None):
    well_metrics = []
    unique_wells = np.unique(well_indices)
    for well_idx in unique_wells:
        mask = well_indices == well_idx
        well_true, well_pred = y_true[mask], y_pred[mask]
        valid_mask = ~np.isnan(well_true) & ~np.isnan(well_pred)
        if np.sum(valid_mask) == 0: continue
        wt, wp = well_true[valid_mask], well_pred[valid_mask]
        rmse = np.sqrt(np.mean((wp - wt)**2))
        mae = np.mean(np.abs(wp - wt))
        r2 = 1 - np.sum((wt - wp)**2) / (np.sum((wt - np.mean(wt))**2) + 1e-8)
        mape = np.mean(np.abs((wt - wp) / (wt + 1e-8))) * 100
        metrics = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
        location = f"井_{well_idx+1}" if well_names is None else well_names[well_idx]
        well_metrics.append((well_idx, metrics, location, x_coords[well_idx], y_coords[well_idx]))
    return well_metrics

def save_well_results_to_excel(well_metrics_list, excel_path, model_name="GTWR-PINN"):
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('井位指标')
    cols = ['井位编号', '井位位置', 'X坐标', 'Y坐标', 'R2', 'RMSE', 'MAE', 'MAPE']
    for i, name in enumerate(cols): sheet.write(0, i, name)
    for row_idx, (w_idx, m, loc, x, y) in enumerate(well_metrics_list, 1):
        sheet.write(row_idx, 0, int(w_idx+1))
        sheet.write(row_idx, 1, loc)
        sheet.write(row_idx, 2, float(x))
        sheet.write(row_idx, 3, float(y))
        sheet.write(row_idx, 4, float(m['R2']))
        sheet.write(row_idx, 5, float(m['RMSE']))
        sheet.write(row_idx, 6, float(m['MAE']))
        sheet.write(row_idx, 7, float(m['MAPE']))
    workbook.save(excel_path)

def print_well_metrics_table(well_metrics_list, train_wells, test_wells, model_name="GTWR-PINN"):
    from tabulate import tabulate
    print("\n" + "=" * 120)
    print(f"{model_name:^120}")
    table_data = []
    for well_idx, metrics_w, location, x_pos, y_pos in well_metrics_list:
        dataset_type = "训练集" if well_idx in train_wells else "测试集"
        table_data.append([f"{location} (井 #{well_idx+1})", f"{metrics_w['R2']:.4f}", 
                          f"{metrics_w['RMSE']:.2f}", f"{metrics_w['MAE']:.2f}", 
                          f"{metrics_w['MAPE']:.1f}%", dataset_type])
    table_data.sort(key=lambda x: float(x[1]) if x[1] != "NaN" else -99, reverse=True)
    print(tabulate(table_data, headers=["井位信息", "R²", "RMSE", "MAE", "MAPE", "数据集"], 
                  tablefmt="grid", stralign="right"))
    print("=" * 120)

def export_well_results(H_true, H_pred, well_indices, x_coords, y_coords, train_wells, test_wells, 
                       output_dir, model_name="GTWR-PINN"):
    well_metrics_list = calculate_well_metrics(H_true.flatten(), H_pred.flatten(), well_indices, x_coords, y_coords)
    excel_path = os.path.join(output_dir, f'{model_name}_井位结果.xls')
    save_well_results_to_excel(well_metrics_list, excel_path, model_name)
    print_well_metrics_table(well_metrics_list, train_wells, test_wells, model_name)

if __name__ == '__main__':
    residual_correction_gtwr_pinn()
