"""
TransMICRO-Net figures — SCI journal style (NMI-pastel palette, refined layout)
Subplot labels Arial bold. Export PNG @600dpi + PDF.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

# ---------------------------------------------------------------
# SCI journal style (nature-figure NMI pastel family)
# ---------------------------------------------------------------
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#4D4D4D",
    "xtick.color": "#272727",
    "ytick.color": "#272727",
    "axes.labelcolor": "#272727",
})

# NMI pastel unified family
BLUE_D   = '#484878'   # baseline dark (Alginate / RF)
BLUE_M   = '#7884B4'   # baseline mid
BLUE_S   = '#B4C0E4'   # baseline soft
OURS_1   = '#E4E4F0'   # ours tiny
OURS_2   = '#E4CCD8'   # ours base
OURS_3   = '#F0C0CC'   # ours large
LILAC    = '#E0E0F0'
AQUA     = '#E0F0F0'
PEACH    = '#F0E0D0'
NEU_L    = '#D8D8D8'
NEU_M    = '#A8A8A8'
NEU_D    = '#606060'
UP_GREEN = '#2E9E44'
DOWN_RED = '#E53935'
GOLD     = '#C9A227'
TEAL     = '#42949E'
VIOLET   = '#9A4D8E'

# Semantic mapping: OHA = blue_dark, Alginate = teal, ADA = violet, Ours = red accent
C_OHA  = BLUE_D
C_ALG  = TEAL
C_ADA  = VIOLET
C_OURS = DOWN_RED
C_BASE = [BLUE_D, BLUE_M, NEU_M]
C_MAT  = [C_OHA, C_ALG]
MAT_NAMES = ['OHA-GEL', 'Alginate']

BASE = r'C:\Users\TS\Desktop\TransMICRO-Net'
DATA_DIR = os.path.join(BASE, 'processed_data')
MODEL_DIR = os.path.join(BASE, 'models')
ANALYSIS_DIR = os.path.join(BASE, 'analysis')
DOWN_DIR = os.path.join(ANALYSIS_DIR, 'downstream')
OUT_DIR = os.path.join(ANALYSIS_DIR, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

def add_label(ax, label, x=-0.20, y=1.03):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight='bold',
            fontfamily='Arial', va='bottom', ha='left')

def set_style(ax, grid=True):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(direction='in', length=3, width=0.7)
    if grid:
        ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.4, color=NEU_M)
        ax.set_axisbelow(True)
    ax.set_facecolor('white')

def save_fig(fig, name, dpi=600):
    fig.savefig(os.path.join(OUT_DIR, name + '.png'), dpi=dpi, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, name + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'{name} saved')

def smooth_curve(curve, window_length=9, polyorder=2):
    from scipy.signal import savgol_filter
    if len(curve) <= window_length:
        return curve
    return savgol_filter(curve, window_length=window_length, polyorder=polyorder, mode='interp')

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
strain = np.load(os.path.join(DATA_DIR, 'strain_grid.npy'))
X_test = np.load(os.path.join(DATA_DIR, 'test_curves.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'test_labels.npy'))

ours = np.load(os.path.join(MODEL_DIR, 'final_test_predictions.npz'))
true_ext = ours['true']
pred_ext = ours['pred']

base_preds = {
    'RF':  np.load(os.path.join(ANALYSIS_DIR, 'baseline_Random Forest (5 trees)_test_preds.npz'))['pred'],
    'DT':  np.load(os.path.join(ANALYSIS_DIR, 'baseline_Decision Tree (max_depth=3)_test_preds.npz'))['pred'],
    'KNN': np.load(os.path.join(ANALYSIS_DIR, 'baseline_KNN (k=5)_test_preds.npz'))['pred'],
}
abl_preds = {
    'No Attention': np.load(os.path.join(ANALYSIS_DIR, 'ablation_No Attention_test_preds.npz'))['pred'],
    'No FiLM':      np.load(os.path.join(ANALYSIS_DIR, 'ablation_No FiLM_test_preds.npz'))['pred'],
}

fold_recon = {}
for fold in range(1, 4):
    d = np.load(os.path.join(MODEL_DIR, f'fold{fold}_val_predictions.npz'))
    rt, rp = d['recon_true'], d['recon_pred']
    r2s = []
    for i in range(len(rt)):
        denom = np.sum((rt[i] - np.mean(rt[i]))**2)
        r2s.append(max(0, 1 - np.sum((rt[i]-rp[i])**2)/denom) if denom > 0 else 0.0)
    fold_recon[fold] = {'r2_mean': float(np.mean(r2s)),
                        'rmse': float(np.sqrt(np.mean((rt - rp)**2)))}

summary = pd.read_csv(os.path.join(ANALYSIS_DIR, 'summary_ranking_final.csv'))
ours_row = summary[summary['Model'] == 'TransMICRO-Net (Ours)'].iloc[0]
physics_folds = [float(ours_row['Physics_Fold1']), float(ours_row['Physics_Fold2']), float(ours_row['Physics_Fold3'])]

targets = ['Peak stress', 'Yield point', 'Toughness']
target_keys = ['peak_stress', 'yield_point', 'toughness']
units = ['Pa', 'Pa', 'J m$^{-3}$']

mae = {
    'Ours':      {'peak': 65.28, 'yield': 0.162, 'tough': 2.32, 'overall': 22.59},
    'RF':        {'peak': 34.35, 'yield': 0.303, 'tough': 5.78, 'overall': 13.48},
    'DT':        {'peak': 69.39, 'yield': 0.196, 'tough': 3.71, 'overall': 24.43},
    'KNN':       {'peak': 45.48, 'yield': 0.879, 'tough': 3.94, 'overall': 16.76},
    'No Attn':   {'peak': 112.45, 'yield': 0.989, 'tough': 1.82, 'overall': 38.42},
    'No FiLM':   {'peak': 74.24, 'yield': 0.425, 'tough': 2.73, 'overall': 25.80},
}
r2_ext = {'Ours': 0.896, 'No Attn': 0.655, 'No FiLM': 0.831}
print('COMMON READY')
