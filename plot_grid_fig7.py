"""SCI-style Fig 7 (2x3 grid): latent + PDP + cycle
Hero: UMAP (A spans left column, tall). B: PDP. C: cycle stability.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sci_common.py'), encoding='utf-8').read())

from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE

DOWN = os.path.join(ANALYSIS_DIR, 'downstream')
latent_train = np.load(os.path.join(DOWN, 'latent_train.npy'))
latent_test = np.load(os.path.join(DOWN, 'latent_test.npy'))
latent_labels_tr = np.load(os.path.join(DOWN, 'latent_labels_train.npy'))
pdp_oha = np.load(os.path.join(DOWN, 'pdp_oha.npz'))

try:
    from umap import UMAP
    reducer = UMAP(n_neighbors=8, min_dist=0.2, random_state=42)
    z_tr = reducer.fit_transform(latent_train)
    z_te = reducer.transform(latent_test)
except Exception:
    reducer = TSNE(n_components=2, random_state=42, perplexity=5)
    z_all = reducer.fit_transform(np.vstack([latent_train, latent_test]))
    z_tr, z_te = z_all[:len(latent_train)], z_all[len(latent_train):]

fig = plt.figure(figsize=(10.0, 5.4))
gs = GridSpec(2, 3, width_ratios=[1.3, 1, 1], hspace=0.55, wspace=0.42)

# HERO A: UMAP
axA = fig.add_subplot(gs[:, 0])
for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
    mask = latent_labels_tr == mat
    axA.scatter(z_tr[mask, 0], z_tr[mask, 1], s=55, c=color, alpha=0.85,
               edgecolors='white', linewidth=0.6, label=name, zorder=3)
axA.scatter(z_te[:, 0], z_te[:, 1], s=85, marker='D', c=C_ADA, alpha=0.95,
           edgecolors='white', linewidth=0.8, label='ADA-GEL (test)', zorder=5)
axA.set_xlabel('UMAP-1'); axA.set_ylabel('UMAP-2')
axA.legend(fontsize=6, loc='best', frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axA)
add_label(axA, 'A', x=-0.18, y=1.05)
axA.set_title('Hero: latent space', fontsize=9, fontweight='bold')

# B: PDP peak stress
axB = fig.add_subplot(gs[0, 1])
conc = pdp_oha['conc']
preds = pdp_oha['preds']
pd_vals = preds[:, 0] if preds.ndim == 2 else preds
axB.plot(conc, pd_vals, '-o', color=C_OURS, lw=1.7, ms=4, mfc='white', mew=1.2)
axB.set_xlabel('Concentration (%)')
axB.set_ylabel('Predicted peak stress (Pa)')
set_style(axB); add_label(axB, 'B', x=-0.14, y=1.06)

# C: cycle stability
csv = os.path.join(DOWN, 'cycle_stability.csv')
axC = fig.add_subplot(gs[0, 2])
if os.path.exists(csv):
    cd = pd.read_csv(csv)
    for (mat, conc1), g in cd.groupby(['material', 'conc1']):
        pred = dict(zip(g['cycle'], g['pred_peak_stress']))
        if 1 in pred and 3 in pred:
            color = C_OHA if mat == 'OHA-GEL' else C_ALG
            marker = 'o' if mat == 'OHA-GEL' else 's'
            axC.scatter(pred[1], pred[3], s=60, c=color, marker=marker,
                       edgecolors='white', linewidth=0.6, alpha=0.9,
                       label=f'{mat} {conc1}%' if conc1 in (2.5, 3.75) else None)
    axC.plot([100, 1200], [100, 1200], 'k--', lw=0.8, alpha=0.45)
    axC.set_xlim(100, 1200); axC.set_ylim(100, 1200)
    axC.set_xlabel('Cycle 1 peak stress (Pa)')
    axC.set_ylabel('Cycle 3 peak stress (Pa)')
    axC.legend(fontsize=5, loc='best', frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axC); add_label(axC, 'C', x=-0.14, y=1.06)

# D: error distribution by material (bottom-left of right column)
axD = fig.add_subplot(gs[1, 1])
gs_recon = np.load(os.path.join(DOWN, 'recon_error_diagnosis.npz'))
errs = gs_recon['errors']
err_mats = gs_recon['materials']
err_mag = np.abs(errs).mean(axis=1)
for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
    mask = err_mats == mat
    vals = err_mag[mask]
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
    axD.scatter(np.full(len(vals), 0)+jitter, vals, s=34, c=color, alpha=0.85,
                edgecolors='white', linewidth=0.6, label=name, zorder=3)
    axD.hlines(np.median(vals), -0.25, 0.25, color=color, lw=1.6, zorder=4)
axD.set_xticks([0, 1]); axD.set_xticklabels(['OHA-GEL', 'Alginate'], fontsize=7)
axD.set_ylabel('Mean |recon error|')
axD.legend(fontsize=6, loc='upper right', frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axD); add_label(axD, 'D', x=-0.14, y=1.06)

# E: latent space with legend separate (right column bottom)
axE = fig.add_subplot(gs[1, 2])
# latent distance between materials
from scipy.spatial.distance import cdist
o = z_tr[latent_labels_tr == 0]; a = z_tr[latent_labels_tr == 1]
d_oo = cdist(o, o).mean(); d_aa = cdist(a, a).mean(); d_oa = cdist(o, a).mean()
mat_lab = ['OHA-OHA', 'Alg-Alg', 'OHA-Alg']
mat_val = [d_oo, d_aa, d_oa]
bars = axE.bar(mat_lab, mat_val, color=[C_OHA, C_ALG, C_ADA], alpha=0.9,
              width=0.55, edgecolor='white', linewidth=0.8)
for b, v in zip(bars, mat_val):
    axE.text(b.get_x()+b.get_width()/2, v+0.05, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
axE.set_ylabel('Mean latent distance')
set_style(axE); add_label(axE, 'E', x=-0.14, y=1.06)

plt.tight_layout()
save_fig(fig, 'Fig7')