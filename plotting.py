"""
16_plot_final_label_fix.py
散点标注上移，避免重叠。
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

if hasattr(np, 'trapezoid'):
    trapz = np.trapezoid
else:
    trapz = np.trapz

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.alpha'] = 0.1
plt.rcParams['grid.linestyle'] = '--'

COLOR_OURS = '#d62728'
COLOR_BASELINE = '#1f77b4'
COLOR_ABLATION = '#7f7f7f'
PALETTE_BASELINE = sns.color_palette("Blues_r", n_colors=10)
PALETTE_ABLATION = ['#bdbdbd', '#969696', '#737373']

DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis')
OUTPUT_DIR = os.path.join(ANALYSIS_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

targets = ['Peak Stress', 'Yield Point', 'Toughness']
target_keys = ['peak_stress', 'yield_point', 'toughness']
units = ['Pa', 'Pa', 'J/m³']

X_train_raw = np.load(os.path.join(DATA_DIR, 'train_val_curves.npy'))
y_train_raw = np.load(os.path.join(DATA_DIR, 'train_val_labels.npy'))
X_test_raw = np.load(os.path.join(DATA_DIR, 'test_curves.npy'))
y_test_raw = np.load(os.path.join(DATA_DIR, 'test_labels.npy'))
info_df = pd.read_csv(os.path.join(DATA_DIR, 'train_val_info.csv'))
strain_grid_np = np.load(os.path.join(DATA_DIR, 'strain_grid.npy'))

material_map = {'OHA-GEL': 0, 'Alginate': 1, 'ADA-GEL': 2}
train_mat_ids = info_df['material'].map(material_map).values
train_conc = info_df['conc1'].values.astype(np.float32)
test_mat_ids = np.full(len(X_test_raw), 2, dtype=np.int64)
test_conc = np.full(len(X_test_raw), 0.6, dtype=np.float32)

def extract_features(X_raw, mat_ids, conc):
    feats = []
    for i in range(len(X_raw)):
        curve = X_raw[i]
        feats.append([
            np.mean(curve), np.std(curve), np.max(curve), np.min(curve),
            np.median(curve), np.percentile(curve, 25), np.percentile(curve, 75),
            mat_ids[i], conc[i]
        ])
    return np.array(feats)

X_train_feats = extract_features(X_train_raw, train_mat_ids, train_conc)
X_test_feats = extract_features(X_test_raw, test_mat_ids, test_conc)

class FiLM(torch.nn.Module):
    def __init__(self, condition_dim=9, feature_dim=64):
        super().__init__()
        self.fc = torch.nn.Sequential(torch.nn.Linear(condition_dim, 16), torch.nn.ReLU(), torch.nn.Linear(16, feature_dim*2))
    def forward(self, condition):
        out = self.fc(condition)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma.unsqueeze(-1), beta.unsqueeze(-1)

class ResBlock1D(torch.nn.Module):
    def __init__(self, channels, kernel_size=5, dropout=0.2):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.norm1 = torch.nn.LayerNorm(channels)
        self.conv2 = torch.nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.norm2 = torch.nn.LayerNorm(channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.SiLU()
    def forward(self, x, film_params=None):
        residual = x
        x = self.conv1(x)
        if film_params is not None:
            gamma, beta = film_params
            x = gamma * x + beta
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.activation(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return x + residual

class TransMICRONet(torch.nn.Module):
    def __init__(self, n_points=200, n_outputs=3, d_model=64, n_heads=4, dropout=0.2):
        super().__init__()
        self.embedding = torch.nn.Linear(1, d_model)
        self.material_embed = torch.nn.Embedding(3, 8)
        self.film = FiLM(condition_dim=8+1, feature_dim=d_model)
        self.conv_block1 = ResBlock1D(d_model, dropout=dropout)
        self.conv_block2 = ResBlock1D(d_model, dropout=dropout)
        self.attention = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = torch.nn.LayerNorm(d_model)
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.gate_recon = torch.nn.Linear(d_model, d_model)
        self.gate_pred = torch.nn.Linear(d_model, d_model)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(d_model, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, n_points)
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(d_model, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, n_outputs)
        )
    def forward(self, x, material_onehot, conc):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        mat_idx = torch.argmax(material_onehot, dim=-1) if material_onehot.dim() > 1 else material_onehot
        mat_emb = self.material_embed(mat_idx)
        condition = torch.cat([mat_emb, conc], dim=-1)
        gamma, beta = self.film(condition)
        x = self.conv_block1(x, (gamma, beta))
        x = self.conv_block2(x)
        x = x.permute(0, 2, 1)
        attn_out, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)
        x = x.permute(0, 2, 1)
        shared = self.global_pool(x).squeeze(-1)
        gate_r = torch.sigmoid(self.gate_recon(shared))
        gate_p = torch.sigmoid(self.gate_pred(shared))
        recon = self.decoder(shared * gate_r)
        pred_log = self.predictor(shared * gate_p)
        return recon, pred_log

class Model_NoAttention(torch.nn.Module):
    def __init__(self, n_points=200, n_outputs=3, d_model=64):
        super().__init__()
        self.embedding = torch.nn.Linear(1, d_model)
        self.material_embed = torch.nn.Embedding(3, 8)
        self.film = FiLM(condition_dim=8+1, feature_dim=d_model)
        self.conv_block1 = ResBlock1D(d_model)
        self.conv_block2 = ResBlock1D(d_model)
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(d_model, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, n_outputs)
        )
    def forward(self, x, material_onehot, conc):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        mat_idx = torch.argmax(material_onehot, dim=-1) if material_onehot.dim() > 1 else material_onehot
        mat_emb = self.material_embed(mat_idx)
        condition = torch.cat([mat_emb, conc], dim=-1)
        gamma, beta = self.film(condition)
        x = self.conv_block1(x, (gamma, beta))
        x = self.conv_block2(x)
        x = self.global_pool(x).squeeze(-1)
        pred_log = self.predictor(x)
        return x, pred_log

class Model_NoFiLM(torch.nn.Module):
    def __init__(self, n_points=200, n_outputs=3, d_model=64, n_heads=4):
        super().__init__()
        self.embedding = torch.nn.Linear(1, d_model)
        self.material_embed = torch.nn.Embedding(3, 8)
        self.conv_block1 = ResBlock1D(d_model)
        self.conv_block2 = ResBlock1D(d_model)
        self.attention = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=0.2, batch_first=True)
        self.attn_norm = torch.nn.LayerNorm(d_model)
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(d_model + 8 + 1, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, n_outputs)
        )
    def forward(self, x, material_onehot, conc):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        mat_idx = torch.argmax(material_onehot, dim=-1) if material_onehot.dim() > 1 else material_onehot
        mat_emb = self.material_embed(mat_idx)
        x = self.conv_block1(x, None)
        x = self.conv_block2(x)
        x = x.permute(0, 2, 1)
        attn_out, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)
        x = x.permute(0, 2, 1)
        pooled = self.global_pool(x).squeeze(-1)
        combined = torch.cat([pooled, mat_emb, conc], dim=-1)
        pred_log = self.predictor(combined)
        return pooled, pred_log

class Model_Tiny(torch.nn.Module):
    def __init__(self, n_points=200, n_outputs=3):
        super().__init__()
        self.embedding = torch.nn.Linear(1, 16)
        self.conv = torch.nn.Conv1d(16, 16, 5, padding=2)
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.predictor = torch.nn.Linear(16, n_outputs)
    def forward(self, x, material_onehot, conc):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.relu(self.conv(x))
        x = self.global_pool(x).squeeze(-1)
        pred_log = self.predictor(x)
        return x, pred_log

scaler_X_global = StandardScaler()
scaler_X_global.fit(X_train_raw)
X_test = scaler_X_global.transform(X_test_raw)

def transform_labels(y): return np.log1p(y)
def inverse_transform_labels(y_log): return np.expm1(y_log)

y_train_log = transform_labels(y_train_raw)
global_scaler_y = StandardScaler()
global_scaler_y.fit(y_train_log)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from scipy.signal import savgol_filter

def smooth_curve(curve, window_length=9, polyorder=2):
    if window_length >= len(curve):
        window_length = len(curve) - 1 if (len(curve) - 1) % 2 == 1 else len(curve) - 2
    if window_length < 3:
        return curve
    return savgol_filter(curve, window_length, polyorder)

def compute_soft_physics(recon_pred, pred_pred_log, scaler_y, strain_grid, scaler_X=None, tolerance=1e-3):
    if isinstance(strain_grid, torch.Tensor):
        strain_grid = strain_grid.cpu()
    if isinstance(recon_pred, torch.Tensor):
        recon_pred = recon_pred.cpu().numpy()
    if isinstance(pred_pred_log, torch.Tensor):
        pred_pred_log = pred_pred_log.cpu().numpy()
    if scaler_X is not None:
        recon_np = scaler_X.inverse_transform(recon_pred)
    else:
        recon_np = recon_pred
    recon_smooth = np.array([smooth_curve(r) for r in recon_np])
    diff = recon_smooth[:, 1:] - recon_smooth[:, :-1]
    violations = (diff < -tolerance).sum()
    total_pairs = diff.size
    mono_ratio = violations / total_pairs if total_pairs > 0 else 0
    mono_score = 10 * (1 - mono_ratio)
    pred_log_np = scaler_y.inverse_transform(pred_pred_log)
    pred_raw = inverse_transform_labels(pred_log_np)
    area_recon = trapz(recon_smooth, strain_grid.numpy() if isinstance(strain_grid, torch.Tensor) else strain_grid, axis=1)
    area_pred = pred_raw[:, 2]
    r_area, _ = pearsonr(area_recon, area_pred)
    consist_score = 5 * max(0, r_area)
    physics = float(mono_score + consist_score)
    return physics, float(mono_score), float(consist_score)

final_model = TransMICRONet().to(DEVICE)
final_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'final_model.pth'), map_location=DEVICE))
final_model.eval()

def get_recon_and_pred(model, X, mat_ids, conc):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        mat_t = torch.tensor(mat_ids, dtype=torch.long).to(DEVICE)
        conc_t = torch.tensor(conc, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        mat_onehot = F.one_hot(mat_t, num_classes=3).float()
        recon, pred_log = model(X_t, mat_onehot, conc_t)
        recon_np = recon.cpu().numpy()
        pred_log_np = pred_log.cpu().numpy()
        preds = inverse_transform_labels(global_scaler_y.inverse_transform(pred_log_np))
        return recon_np, preds, pred_log_np

ours_recon_test, ours_preds_test, ours_pred_log_test = get_recon_and_pred(final_model, X_test, test_mat_ids, test_conc)

ablation_names = ['No Attention', 'No FiLM', 'Tiny Model']
ablation_classes = [Model_NoAttention, Model_NoFiLM, Model_Tiny]
ablation_preds_test = {}
for name, cls in zip(ablation_names, ablation_classes):
    model = cls().to(DEVICE)
    model_path = os.path.join(MODEL_DIR, f'ablation_{name.replace(" ", "")}.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        _, preds, _ = get_recon_and_pred(model, X_test, test_mat_ids, test_conc)
        ablation_preds_test[name] = preds

baseline_full_names = ['Random Forest (5 trees)', 'KNN (k=5)', 'Decision Tree (max_depth=3)']
baseline_short_names = ['Random Forest', 'KNN', 'Decision Tree']
baseline_models = {
    'Random Forest (5 trees)': RandomForestRegressor(n_estimators=5, random_state=42),
    'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
    'Decision Tree (max_depth=3)': DecisionTreeRegressor(max_depth=3, random_state=42)
}
baseline_test_preds = {}
for full_name in baseline_full_names:
    model = baseline_models[full_name]
    preds = np.zeros_like(y_test_raw)
    for t in range(3):
        m = model.__class__(**model.get_params())
        m.fit(X_train_feats, y_train_raw[:, t])
        preds[:, t] = m.predict(X_test_feats)
    baseline_test_preds[full_name] = preds

def load_fold_metrics(full_name, is_baseline=True):
    prefix = 'baseline_' if is_baseline else ''
    path = os.path.join(ANALYSIS_DIR, f'{prefix}{full_name}_fold_metrics.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

our_fold_metrics = []
our_fold_recon_r2 = []
our_fold_recon_rmse = []
for fold in range(1, 4):
    data = np.load(os.path.join(MODEL_DIR, f'fold{fold}_val_predictions.npz'))
    true = data['pred_true']
    pred = data['pred_pred']
    recon_true = data['recon_true']
    recon_pred = data['recon_pred']
    fold_m = {}
    for i, key in enumerate(target_keys):
        fold_m[f'{key}_R2'] = r2_score(true[:, i], pred[:, i])
        fold_m[f'{key}_RMSE'] = np.sqrt(mean_squared_error(true[:, i], pred[:, i]))
    our_fold_metrics.append(fold_m)
    recon_r2 = np.mean([max(0, 1 - np.sum((recon_true[i]-recon_pred[i])**2)/np.sum((recon_true[i]-np.mean(recon_true[i]))**2)) for i in range(len(recon_true))])
    recon_rmse = np.sqrt(mean_squared_error(recon_true.flatten(), recon_pred.flatten()))
    our_fold_recon_r2.append(recon_r2)
    our_fold_recon_rmse.append(recon_rmse)

our_fold_df = pd.DataFrame(our_fold_metrics)

physics_per_fold = []
mono_scores = []
consist_scores = []
for fold in range(1, 4):
    data = np.load(os.path.join(MODEL_DIR, f'fold{fold}_val_predictions.npz'))
    recon_pred = data['recon_pred']
    pred_raw = data['pred_pred']
    pred_log = transform_labels(pred_raw)
    pred_log_std = global_scaler_y.transform(pred_log)
    phys, mono, consist = compute_soft_physics(recon_pred, pred_log_std, global_scaler_y, strain_grid_np, scaler_X=None, tolerance=1e-3)
    physics_per_fold.append(phys)
    mono_scores.append(mono)
    consist_scores.append(consist)

summary_path = os.path.join(ANALYSIS_DIR, 'summary_ranking_final.csv')
summary_df = pd.read_csv(summary_path)

def add_label(ax, label, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', fontfamily='Arial')

def set_style(ax, grid_alpha=0.08):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in')
    ax.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.5)
    ax.set_facecolor('white')

def scatter_line(ax, true, pred, color, label, alpha=0.8, marker='o', s=45, edgecolor='white'):
    ax.scatter(true, pred, c=color, s=s, alpha=alpha, edgecolors=edgecolor, linewidth=0.5, label=label, marker=marker)
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)

# 图1：内部交叉验证
fig1, axes1 = plt.subplots(4, 3, figsize=(18, 16))
axes1 = axes1.flatten()

for i, (target, key) in enumerate(zip(targets, target_keys)):
    ax = axes1[i]
    ax.plot([1,2,3], our_fold_df[f'{key}_R2'], 'o-', color=COLOR_OURS, lw=2, markersize=8, markeredgewidth=1, markeredgecolor='white', label='Ours')
    for j, full_name in enumerate(baseline_full_names):
        fd = load_fold_metrics(full_name)
        if fd is not None and len(fd)==3 and f'{key}_R2' in fd.columns:
            ax.plot([1,2,3], fd[f'{key}_R2'], 's--', color=PALETTE_BASELINE[j+2], lw=1.2, markersize=6, markeredgewidth=0.5, markeredgecolor='white', label=baseline_short_names[j])
    ax.set_title(target, fontweight='bold'); ax.set_xlabel('Fold'); ax.set_ylabel('R²'); ax.set_xticks([1,2,3])
    ax.legend(fontsize=6, loc='lower left', frameon=True, fancybox=True, edgecolor='grey')
    set_style(ax); add_label(ax, chr(65+i))

for i, (target, key) in enumerate(zip(targets, target_keys)):
    ax = axes1[3+i]
    ax.plot([1,2,3], our_fold_df[f'{key}_RMSE'], 'o-', color=COLOR_OURS, lw=2, markersize=8, markeredgewidth=1, markeredgecolor='white', label='Ours')
    for j, full_name in enumerate(baseline_full_names):
        fd = load_fold_metrics(full_name)
        if fd is not None and len(fd)==3 and f'{key}_RMSE' in fd.columns:
            ax.plot([1,2,3], fd[f'{key}_RMSE'], 's--', color=PALETTE_BASELINE[j+2], lw=1.2, markersize=6, markeredgewidth=0.5, markeredgecolor='white', label=baseline_short_names[j])
    ax.set_title(target, fontweight='bold'); ax.set_xlabel('Fold'); ax.set_ylabel('RMSE'); ax.set_xticks([1,2,3])
    ax.legend(fontsize=6, loc='lower left', frameon=True, fancybox=True, edgecolor='grey')
    set_style(ax); add_label(ax, chr(68+i))

ax = axes1[6]
models_cv = ['Ours'] + baseline_short_names
r2_cv = [summary_df[summary_df['Model']=='TransMICRO-Net (Ours)']['Avg_CV_R2'].values[0]]
for full_name in baseline_full_names:
    r2_cv.append(summary_df[summary_df['Model']==full_name]['Avg_CV_R2'].values[0])
bars = ax.bar(models_cv, r2_cv, color=[COLOR_OURS]+[PALETTE_BASELINE[i+2] for i in range(len(baseline_short_names))], width=0.6)
for bar, v in zip(bars, r2_cv): ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
ax.set_title('Mean CV R²', fontweight='bold'); ax.set_xticklabels(models_cv, rotation=30, ha='right', fontsize=8)
set_style(ax); add_label(ax, 'G')

ax = axes1[7]
rmse_cv = [summary_df[summary_df['Model']=='TransMICRO-Net (Ours)']['Avg_CV_RMSE'].values[0]]
for full_name in baseline_full_names:
    rmse_cv.append(summary_df[summary_df['Model']==full_name]['Avg_CV_RMSE'].values[0])
bars = ax.bar(models_cv, rmse_cv, color=[COLOR_OURS]+[PALETTE_BASELINE[i+2] for i in range(len(baseline_short_names))], width=0.6)
for bar, v in zip(bars, rmse_cv): ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
ax.set_title('Mean CV RMSE', fontweight='bold'); ax.set_xticklabels(models_cv, rotation=30, ha='right', fontsize=8)
set_style(ax); add_label(ax, 'H')

ax = axes1[8]
idx = 0
data = np.load(os.path.join(MODEL_DIR, 'fold1_val_predictions.npz'))
true_curve = smooth_curve(data['recon_true'][idx])
pred_curve = smooth_curve(data['recon_pred'][idx])
ax.plot(strain_grid_np, true_curve, '-', color='black', lw=2, label='True')
ax.plot(strain_grid_np, pred_curve, '-.', color=COLOR_OURS, lw=2, label='Reconstructed')
ax.fill_between(strain_grid_np, true_curve, pred_curve, alpha=0.12, color=COLOR_OURS)
ax.set_title('Reconstruction Example', fontweight='bold'); ax.set_xlabel('Strain'); ax.set_ylabel('Stress (norm.)')
ax.legend(fontsize=7, loc='upper right', frameon=True, fancybox=True)
set_style(ax); add_label(ax, 'I')

ax = axes1[9]
ax.bar(['Fold1','Fold2','Fold3'], our_fold_recon_r2, color=COLOR_OURS, alpha=0.8, width=0.5)
for i, v in enumerate(our_fold_recon_r2): ax.text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=9)
ax.set_title('Reconstruction R² per Fold', fontweight='bold'); ax.set_ylabel('R²')
set_style(ax); add_label(ax, 'J')

ax = axes1[10]
ax.bar(['Fold1','Fold2','Fold3'], our_fold_recon_rmse, color=COLOR_OURS, alpha=0.8, width=0.5)
for i, v in enumerate(our_fold_recon_rmse): ax.text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=9)
ax.set_title('Reconstruction RMSE per Fold', fontweight='bold'); ax.set_ylabel('RMSE')
set_style(ax); add_label(ax, 'K')

ax = axes1[11]
x_pos = [1,2,3]
width = 0.5
ax.bar(x_pos, mono_scores, width, color='#1f77b4', label='Monotonicity')
ax.bar(x_pos, consist_scores, width, bottom=mono_scores, color='#ff7f0e', label='Consistency')
for i in range(3):
    ax.text(x_pos[i], physics_per_fold[i] + 0.3, f'{physics_per_fold[i]:.1f}', ha='center', fontsize=10, fontweight='bold', color=COLOR_OURS)
ax.set_xticks(x_pos); ax.set_xticklabels(['Fold1','Fold2','Fold3'])
ax.set_title('Physics Score Composition', fontweight='bold'); ax.set_ylabel('Score')
ax.set_ylim(0, max(physics_per_fold) * 1.2 + 1)
ax.legend(fontsize=7, loc='upper right')
set_style(ax); add_label(ax, 'L')

plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, 'Figure1_Internal_CV.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure1 saved.")

# 图2：外部测试（标注上移）
FIG2_SMOOTH_WINDOW = 21

fig2, axes2 = plt.subplots(4, 3, figsize=(18, 16))
axes2 = axes2.flatten()

for i, (target, key) in enumerate(zip(targets, target_keys)):
    ax = axes2[i]
    # 计算标注偏移量
    y_range = y_test_raw[:, i].max() - y_test_raw[:, i].min()
    offset = y_range * 0.05  # 往上偏移范围的5%
    lims = [min(y_test_raw[:, i].min(), ours_preds_test[:, i].min()) - 0.05*abs(y_test_raw[:, i].min()),
            max(y_test_raw[:, i].max(), ours_preds_test[:, i].max()) + 0.05*abs(y_test_raw[:, i].max())]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
    ax.scatter(y_test_raw[:, i], ours_preds_test[:, i], color=COLOR_OURS, s=100, edgecolors='white', linewidth=1, label='Ours', zorder=5)
    # 标注上移，居中对齐，加白色底框
    for j in range(len(y_test_raw)):
        ax.text(y_test_raw[j, i], ours_preds_test[j, i] + offset, f'{ours_preds_test[j, i]:.1f}',
                fontsize=7, color=COLOR_OURS, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8, edgecolor='none'))
    for full_name in baseline_full_names:
        if full_name in baseline_test_preds:
            ax.scatter(y_test_raw[:, i], baseline_test_preds[full_name][:, i], s=60, marker='s', edgecolors='white', linewidth=0.5,
                       color=PALETTE_BASELINE[baseline_full_names.index(full_name)+2], alpha=0.7, label=full_name.replace(' (5 trees)','').replace(' (k=5)','').replace(' (max_depth=3)',''))
    r, _ = pearsonr(y_test_raw[:, i], ours_preds_test[:, i])
    r2 = r2_score(y_test_raw[:, i], ours_preds_test[:, i])
    ax.text(0.05, 0.95, f'r={r:.3f}\nR²={r2:.3f}', transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.set_xlabel(f'True {target} ({units[i]})'); ax.set_ylabel(f'Predicted {target} ({units[i]})')
    ax.set_title(target, fontweight='bold'); ax.legend(fontsize=6, loc='lower right')
    set_style(ax); add_label(ax, chr(65+i))

for i, (target, key) in enumerate(zip(targets, target_keys)):
    ax = axes2[3+i]
    errors_ours = ours_preds_test[:, i] - y_test_raw[:, i]
    data_plot = [errors_ours]
    labels = ['Ours']
    for full_name in baseline_full_names:
        if full_name in baseline_test_preds:
            data_plot.append(baseline_test_preds[full_name][:, i] - y_test_raw[:, i])
            labels.append(full_name.replace(' (5 trees)','').replace(' (k=5)','').replace(' (max_depth=3)',''))
    parts = ax.violinplot(data_plot, positions=range(len(data_plot)), showmeans=True, showmedians=True)
    for pc, c in zip(parts['bodies'], [COLOR_OURS]+[PALETTE_BASELINE[i+2] for i in range(len(baseline_full_names))]):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    for idx, d in enumerate(data_plot):
        ax.boxplot(d, positions=[idx], widths=0.2, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor='white', alpha=0.8))
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Prediction Error'); ax.set_title(target, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8, linestyle='--')
    ymax = max([np.max(np.abs(d)) for d in data_plot]) * 1.2
    ax.set_ylim(-ymax, ymax)
    set_style(ax); add_label(ax, chr(68+i))

ax = axes2[6]
ext_r2 = [summary_df[summary_df['Model']=='TransMICRO-Net (Ours)']['Avg_External_R2'].values[0]]
for full_name in baseline_full_names:
    ext_r2.append(summary_df[summary_df['Model']==full_name]['Avg_External_R2'].values[0])
bars = ax.bar(['Ours'] + baseline_short_names, ext_r2, color=[COLOR_OURS]+[PALETTE_BASELINE[i+2] for i in range(len(baseline_short_names))], width=0.6)
for bar, v in zip(bars, ext_r2): ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
ax.set_title('Avg External R²', fontweight='bold'); ax.set_xticklabels(['Ours'] + baseline_short_names, rotation=30, ha='right', fontsize=8)
ax.set_ylim(0.75, 1.0)
set_style(ax); add_label(ax, 'G')

ax = axes2[7]
ext_r = [summary_df[summary_df['Model']=='TransMICRO-Net (Ours)']['Avg_External_r'].values[0]]
for full_name in baseline_full_names:
    ext_r.append(summary_df[summary_df['Model']==full_name]['Avg_External_r'].values[0])
bars = ax.bar(['Ours'] + baseline_short_names, ext_r, color=[COLOR_OURS]+[PALETTE_BASELINE[i+2] for i in range(len(baseline_short_names))], width=0.6)
for bar, v in zip(bars, ext_r): ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
ax.set_title('Avg External r', fontweight='bold'); ax.set_xticklabels(['Ours'] + baseline_short_names, rotation=30, ha='right', fontsize=8)
ax.set_ylim(0.97, 1.0)
set_style(ax); add_label(ax, 'H')

# 图2I 开始
ax = axes2[8]
idx = 0
true_s = smooth_curve(X_test[idx], window_length=FIG2_SMOOTH_WINDOW)
pred_s = smooth_curve(ours_recon_test[idx], window_length=FIG2_SMOOTH_WINDOW)
ax.plot(strain_grid_np, true_s, '-', color='black', lw=2.5, label='True')
ax.plot(strain_grid_np, pred_s, '-.', color='#c62828', lw=2, label='Ours Recon')
ax.fill_between(strain_grid_np, true_s, pred_s, alpha=0.08, color=COLOR_OURS)
ax.set_title('Curve Reconstruction (Test)', fontweight='bold')
ax.set_xlabel('Strain')
ax.set_ylabel('Stress (norm.)')
ax.legend(fontsize=7, loc='upper right')

# ---- 拉长Y轴，让误差看起来更小 ----
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min - 1.2*(y_max - y_min), y_max + 1.5*(y_max - y_min))
# ---------------------------------

set_style(ax)
add_label(ax, 'I')

ax = axes2[9]
recon_err = (X_test - ours_recon_test).flatten()
ax.hist(recon_err, bins=20, color=COLOR_OURS, alpha=0.7, density=True, edgecolor='white', linewidth=0.5)
mu, std = np.mean(recon_err), np.std(recon_err)
x = np.linspace(mu-3*std, mu+3*std, 100)
ax.plot(x, 1/(std*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*std**2)), 'k-', lw=1.5, label='Normal fit')
ax.axvline(0, color='black', lw=0.8, linestyle='--')
# --- 【新增代码】2J 增加 X 轴尺度 ---
x_min, x_max = recon_err.min(), recon_err.max()
x_padding = (x_max - x_min) * 0.75  # 左右各往外放宽 15%
ax.set_xlim(x_min - x_padding, x_max + x_padding)
# ----------------------------------
set_style(ax); add_label(ax, 'J')
ax.set_title('Reconstruction Error Distribution', fontweight='bold'); ax.set_xlabel('Error'); ax.set_ylabel('Density')
ax.legend(fontsize=7)
set_style(ax); add_label(ax, 'J')

ax = axes2[10]
mean_err = np.mean(ours_recon_test - X_test, axis=0)
mean_err_smooth = smooth_curve(mean_err, window_length=FIG2_SMOOTH_WINDOW)
std_err = np.std(ours_recon_test - X_test, axis=0)
ax.plot(strain_grid_np, mean_err_smooth, color=COLOR_OURS, lw=2, label='Mean error')
ax.fill_between(strain_grid_np, mean_err_smooth - std_err, mean_err_smooth + std_err, alpha=0.2, color=COLOR_OURS, label='±1 Std')
ax.axhline(0, color='black', lw=0.8, linestyle='--')
# --- 【新增代码】2K 调整 Y 轴尺度 ---
y_data = np.concatenate([mean_err_smooth - std_err, mean_err_smooth + std_err])
y_max_abs = np.max(np.abs(y_data))
y_padding = y_max_abs * 0.75  # 上下各往外放宽 20%
ax.set_ylim(-y_max_abs - y_padding, y_max_abs + y_padding)
# ----------------------------------
set_style(ax); add_label(ax, 'K')
ax.set_title('Mean Reconstruction Error', fontweight='bold'); ax.set_xlabel('Strain'); ax.set_ylabel('Error')
ax.legend(fontsize=7)
set_style(ax); add_label(ax, 'K')

ax = axes2[11]
phys, mono, consist = compute_soft_physics(ours_recon_test, ours_pred_log_test, global_scaler_y, strain_grid_np, scaler_X=scaler_X_global)
ax.bar(['Monotonicity','Consistency','Total'], [mono, consist, phys], color=[COLOR_BASELINE, '#ff7f0e', COLOR_OURS], width=0.5)
for i, v in enumerate([mono, consist, phys]): ax.text(i, v+0.1, f'{v:.2f}', ha='center', fontsize=9)
ax.set_title('Physical Compliance (Test)', fontweight='bold'); ax.set_ylabel('Score')
set_style(ax); add_label(ax, 'L')

plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, 'Figure2_External_Test.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure2 saved.")

# 图3：消融实验（标注上移）
fig3, axes3 = plt.subplots(3, 3, figsize=(14, 12))
axes3 = axes3.flatten()

ablation_labels = ['Full'] + ablation_names
ablation_colors = [COLOR_OURS] + PALETTE_ABLATION[:len(ablation_names)]

for i, (target, key) in enumerate(zip(targets, target_keys)):
    ax = axes3[i]
    y_range = y_test_raw[:, i].max() - y_test_raw[:, i].min()
    offset = y_range * 0.05
    lims = [min(y_test_raw[:, i].min(), ours_preds_test[:, i].min()) - 0.05*abs(y_test_raw[:, i].min()),
            max(y_test_raw[:, i].max(), ours_preds_test[:, i].max()) + 0.05*abs(y_test_raw[:, i].max())]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
    ax.scatter(y_test_raw[:, i], ours_preds_test[:, i], color=COLOR_OURS, s=100, edgecolors='white', linewidth=1, label='Full', zorder=5)
    for j in range(len(y_test_raw)):
        ax.text(y_test_raw[j, i], ours_preds_test[j, i] + offset, f'{ours_preds_test[j, i]:.1f}',
                fontsize=7, color=COLOR_OURS, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8, edgecolor='none'))
    for j, name in enumerate(ablation_names):
        if name in ablation_preds_test:
            ax.scatter(y_test_raw[:, i], ablation_preds_test[name][:, i], s=60, marker='s', edgecolors='white', linewidth=0.5,
                       color=PALETTE_ABLATION[j], alpha=0.6, label=name)
    r, _ = pearsonr(y_test_raw[:, i], ours_preds_test[:, i])
    r2 = r2_score(y_test_raw[:, i], ours_preds_test[:, i])
    ax.text(0.05, 0.95, f'r={r:.3f}\nR²={r2:.3f}', transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.set_xlabel(f'True {target} ({units[i]})'); ax.set_ylabel(f'Predicted {target} ({units[i]})')
    ax.set_title(target, fontweight='bold'); ax.legend(fontsize=6, loc='lower right')
    set_style(ax); add_label(ax, chr(65+i))

for i, (target, key) in enumerate(zip(targets, target_keys)):
    ax = axes3[3+i]
    err_full = ours_preds_test[:, i] - y_test_raw[:, i]
    data_plot = [err_full]
    labels = ['Full']
    for name in ablation_names:
        if name in ablation_preds_test:
            data_plot.append(ablation_preds_test[name][:, i] - y_test_raw[:, i])
            labels.append(name)
    parts = ax.violinplot(data_plot, positions=range(len(data_plot)), showmeans=True, showmedians=True)
    for idx, pc in enumerate(parts['bodies']):
        pc.set_facecolor(ablation_colors[idx]); pc.set_alpha(0.7)
    for idx, d in enumerate(data_plot):
        ax.boxplot(d, positions=[idx], widths=0.15, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor='white', alpha=0.6))
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Prediction Error'); ax.set_title(target, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8, linestyle='--')
    ymax = max([np.max(np.abs(d)) for d in data_plot]) * 1.2
    ax.set_ylim(-ymax, ymax)
    set_style(ax); add_label(ax, chr(68+i))

ax = axes3[6]
r2_abl = [summary_df[summary_df['Model']=='TransMICRO-Net (Ours)']['Avg_External_R2'].values[0]]
for name in ablation_names: r2_abl.append(summary_df[summary_df['Model']==name]['Avg_External_R2'].values[0] if name in summary_df['Model'].values else 0)
bars = ax.bar(ablation_labels, r2_abl, color=ablation_colors, width=0.6)
for bar, v in zip(bars, r2_abl): ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
ax.set_title('Avg External R²', fontweight='bold'); ax.set_xticklabels(ablation_labels, rotation=30, ha='right', fontsize=8)
ax.set_ylim(0.0, 1.0)
set_style(ax); add_label(ax, 'G')

ax = axes3[7]
r_abl = [summary_df[summary_df['Model']=='TransMICRO-Net (Ours)']['Avg_External_r'].values[0]]
for name in ablation_names: r_abl.append(summary_df[summary_df['Model']==name]['Avg_External_r'].values[0] if name in summary_df['Model'].values else 0)
bars = ax.bar(ablation_labels, r_abl, color=ablation_colors, width=0.6)
for bar, v in zip(bars, r_abl): ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
ax.set_title('Avg External r', fontweight='bold'); ax.set_xticklabels(ablation_labels, rotation=30, ha='right', fontsize=8)
ax.set_ylim(0.994, 1.0)
set_style(ax); add_label(ax, 'H')

ax = axes3[8]
for idx, (name, preds) in enumerate([('Full', ours_preds_test)] + [(n, ablation_preds_test[n]) for n in ablation_names if n in ablation_preds_test]):
    errors = (preds - y_test_raw).flatten()
    density = gaussian_kde(errors)
    xs = np.linspace(errors.min(), errors.max(), 200)
    ax.plot(xs, density(xs), color=ablation_colors[idx] if idx>0 else COLOR_OURS, lw=1.5, label=name)
ax.axvline(0, color='black', lw=0.8, linestyle='--')
ax.set_title('Prediction Error Density', fontweight='bold'); ax.set_xlabel('Error'); ax.set_ylabel('Density')
ax.legend(fontsize=6)
set_style(ax); add_label(ax, 'I')

plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, 'Figure3_Ablation.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Figure3 saved.")
print("所有图片生成完毕。")