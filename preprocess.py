"""
01_preprocess_from_out.py
从官方 out 净菜筐里提取压缩曲线，生成标准数据集。
"""
import zipfile
import os
import re
import yaml
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import shutil

# ==================== 路径设置 ====================
ZIP_PATH = os.path.join(BASE_DIR, 'out_OHA-GEL_ADA-GEL_Alg.zip')
EXTRACT_DIR = os.path.join(BASE_DIR, 'out_extracted')
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_data')
N_POINTS = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 1. 解压 ====================
if os.path.exists(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)
os.makedirs(EXTRACT_DIR)
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    for member in zf.namelist():
        # 忽略苹果垃圾文件
        if member.startswith('__MACOSX') or '/._' in member or member.endswith('/._'):
            continue
        if member.endswith('/'):
            continue
        target_path = os.path.join(EXTRACT_DIR, member.replace('out_OHA-GEL_ADA-GEL_Alg/', '', 1))
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with zf.open(member) as source, open(target_path, 'wb') as dest:
            shutil.copyfileobj(source, dest)
print("✅ 解压完成")

# ==================== 2. 工具函数 ====================
def parse_material_from_folder(folder_name):
    """从文件夹名提取材料类型和浓度"""
    name = folder_name.lower()
    if 'alg' in name:
        mat = 'Alginate'
        m = re.search(r'(\d+)p(\d+)_alg', name)
        if m:
            return mat, float(f"{m.group(1)}.{m.group(2)}"), None
    elif 'ada-gel' in name or 'ada_gel' in name:
        mat = 'ADA-GEL'
        m = re.search(r'(\d+)p(\d+)_ada', name)
        if m:
            return mat, float(f"{m.group(1)}.{m.group(2)}"), None
    elif 'oha' in name:
        mat = 'OHA-GEL'
        m = re.search(r'gel_(\d+)p(\d+)-(\d+)p(\d+)', name)
        if m:
            c1 = float(f"{m.group(1)}.{m.group(2)}")
            c2 = float(f"{m.group(3)}.{m.group(4)}")
            return mat, c1, c2
    return None, None, None

def process_one_curve(csv_path, yaml_path):
    """处理单条压缩曲线，返回应力应变网格和指标"""
    # 读取几何尺寸
    with open(yaml_path, 'r') as f:
        geom = yaml.safe_load(f)
    # 几何文件可能直接是 {'height': ..., 'radius': ...}
    h = float(geom['height'])
    r = float(geom['radius'])
    area = np.pi * r * r

    # 读取CSV（可能是逗号或空格分隔，尝试自动检测）
    try:
        df = pd.read_csv(csv_path, sep=None, engine='python', comment='#')
    except:
        # 如果失败，手动按空格读
        df = pd.read_csv(csv_path, delim_whitespace=True, comment='#', header=None)
    
    # 如果有表头且列名包含 time/force/displacement，使用列名；否则按索引取前三列
    if 'time' in [c.lower() for c in df.columns]:
        time = df['time'].values
        force = df['normal force'].values if 'normal force' in df.columns else df['force'].values
        disp = df['displacement'].values
    else:
        # 按列位置取
        data = df.values
        time = data[:, 0]
        force = data[:, 1]
        disp = data[:, 2]

    # 压缩位移为负，取绝对值
    disp_abs = np.abs(disp)
    # 排序确保单调递增
    order = np.argsort(disp_abs)
    disp_sorted = disp_abs[order]
    force_sorted = np.abs(force[order])

    if len(disp_sorted) < 10:
        return None

    strain = disp_sorted / h
    stress = force_sorted / area

    # 插值到 N_POINTS
    try:
        f_int = interp1d(strain, stress, kind='linear', bounds_error=False, fill_value='extrapolate')
        strain_grid = np.linspace(strain.min(), strain.max(), N_POINTS)
        stress_grid = f_int(strain_grid)
    except:
        return None

    if np.isnan(stress_grid).any():
        return None

    peak = np.max(stress_grid)
    toughness = np.trapezoid(stress_grid, strain_grid) if hasattr(np, 'trapezoid') else np.trapz(stress_grid, strain_grid)
    n_lin = max(2, int(N_POINTS * 0.2))
    slope, intercept = np.polyfit(strain_grid[:n_lin], stress_grid[:n_lin], 1)
    offset_line = slope * (strain_grid - 0.002) + intercept
    diff = stress_grid - offset_line
    yield_point = stress_grid[np.where(diff >= 0)[0][0]] if np.any(diff >= 0) else peak * 0.8

    return {
        'strain': strain_grid,
        'stress': stress_grid,
        'peak_stress': peak,
        'yield_point': yield_point,
        'toughness': toughness,
        'height': h,
        'radius': r
    }

# ==================== 3. 遍历所有材料文件夹 ====================
all_samples = []
data_root = EXTRACT_DIR

for folder_name in os.listdir(data_root):
    folder_path = os.path.join(data_root, folder_name)
    if not os.path.isdir(folder_path) or folder_name.startswith('.') or folder_name.startswith('__'):
        continue

    mat, c1, c2 = parse_material_from_folder(folder_name)
    if mat is None:
        print(f"⚠️ 跳过无法识别的文件夹: {folder_name}")
        continue

    # 定位 avg 目录
    avg_dir = os.path.join(folder_path, 'processed_averaged_hyperelast', 'avg')
    if not os.path.exists(avg_dir):
        print(f"⚠️ 没有 avg 目录: {avg_dir}")
        continue

    # 几何文件
    yaml_path = os.path.join(avg_dir, 'geometry.yaml')
    if not os.path.exists(yaml_path):
        print(f"⚠️ 缺少 geometry.yaml: {yaml_path}")
        continue

    # 找到所有 _neg_mean.csv 文件（忽略 ._ 开头的）
    csv_files = [f for f in os.listdir(avg_dir) if '_neg_mean.csv' in f and not f.startswith('._')]
    for csv_file in csv_files:
        csv_path = os.path.join(avg_dir, csv_file)
        res = process_one_curve(csv_path, yaml_path)
        if res is None:
            print(f"❌ 失败: {mat} {csv_file}")
            continue

        # 从文件名提取循环编号 (c1, c2, c3)
        cycle = 'unknown'
        match = re.search(r'c(\d)_neg', csv_file)
        if match:
            cycle = f'c{match.group(1)}'

        sample = {
            'material': mat,
            'conc1': c1,
            'conc2': c2,
            'strain': res['strain'],
            'stress': res['stress'],
            'peak_stress': res['peak_stress'],
            'yield_point': res['yield_point'],
            'toughness': res['toughness'],
            'height': res['height'],
            'radius': res['radius'],
            'source_file': csv_file,
            'folder': folder_name,
            'cycle': cycle
        }
        all_samples.append(sample)
        print(f"✅ {mat} {cycle} {csv_file}")

# ==================== 4. 划分数据集 ====================
test_samples = [s for s in all_samples if s['material'] == 'ADA-GEL']
train_val_samples = [s for s in all_samples if s['material'] != 'ADA-GEL']

print(f"\n练习册样本数: {len(train_val_samples)}")
print(f"考题样本数: {len(test_samples)}")

if len(train_val_samples) < 5 or len(test_samples) < 1:
    print("样本太少，无法继续。")
else:
    X_train_val = np.array([s['stress'] for s in train_val_samples])
    y_train_val = np.array([[s['peak_stress'], s['yield_point'], s['toughness']] for s in train_val_samples])
    X_test = np.array([s['stress'] for s in test_samples])
    y_test = np.array([[s['peak_stress'], s['yield_point'], s['toughness']] for s in test_samples])

    np.save(os.path.join(OUTPUT_DIR, 'train_val_curves.npy'), X_train_val)
    np.save(os.path.join(OUTPUT_DIR, 'train_val_labels.npy'), y_train_val)
    np.save(os.path.join(OUTPUT_DIR, 'test_curves.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'test_labels.npy'), y_test)
    pd.DataFrame(train_val_samples).to_csv(os.path.join(OUTPUT_DIR, 'train_val_info.csv'), index=False)
    pd.DataFrame(test_samples).to_csv(os.path.join(OUTPUT_DIR, 'test_info.csv'), index=False)
    np.save(os.path.join(OUTPUT_DIR, 'strain_grid.npy'), train_val_samples[0]['strain'])
    print(f"✅ 全部搞定！干净数据在: {OUTPUT_DIR}")