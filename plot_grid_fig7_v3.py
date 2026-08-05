"""SCI-style Fig 7 v3: 3x3 GRID (9 panels, NO empty slot) - latent, PDP, cycle diagnostics
Upgrades vs v2:
  F: PDP for Alginate (peak stress)                                        [NEW]
  G: cycle 1->2->3 trajectory lines per formulation                        [NEW: advanced]
  H: OHA-GEL PDP for all three indicators (normalised multi-line)          [NEW: advanced]
  I: latent pairwise distance matrix heatmap (3 materials)                 [NEW: advanced]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_common.py'), encoding='utf-8').read())

from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE

DOWN = os.path.join(ANALYSIS_DIR, 'downstream')
latent_train = np.load(os.path.join(DOWN, 'latent_train.npy'))
latent_test = np.load(os.path.join(DOWN, 'latent_test.npy'))
latent_labels_tr = np.load(os.path.join(DOWN, 'latent_labels_train.npy'))
latent_labels_te = np.load(os.path.join(DOWN, 'latent_labels_test.npy'))
pdp_oha = np.load(os.path.join(DOWN, 'pdp_oha.npz'))
pdp_alg = np.load(os.path.join(DOWN, 'pdp_alg.npz'))

try:
    from umap import UMAP
    reducer = UMAP(n_neighbors=8, min_dist=0.2, random_state=42)
    z_tr = reducer.fit_transform(latent_train)
    z_te = reducer.transform(latent_test)
except Exception:
    reducer = TSNE(n_components=2, random_state=42, perplexity=5)
    z_all = reducer.fit_transform(np.vstack([latent_train, latent_test]))
    z_tr, z_te = z_all[:len(latent_train)], z_all[len(latent_train):]

fig = plt.figure(figsize=(10.4, 8.6))
gs = GridSpec(3, 3, hspace=0.62, wspace=0.55)

# ---- A (HERO): UMAP latent space ----
axA = fig.add_subplot(gs[0, 0])
for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
    mask = latent_labels_tr == mat
    axA.scatter(z_tr[mask, 0], z_tr[mask, 1], s=55, c=color, alpha=0.85,
                edgecolors='white', linewidth=0.6, label=name, zorder=3)
axA.scatter(z_te[:, 0], z_te[:, 1], s=85, marker='D', c=C_ADA, alpha=0.95,
            edgecolors='white', linewidth=0.8, label='ADA-GEL (test)', zorder=5)
axA.set_xlabel('UMAP-1'); axA.set_ylabel('UMAP-2')
axA.legend(fontsize=6.5, loc='best', frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
axA.set_title('Hero: latent space', fontsize=9, fontweight='bold')
set_style(axA); add_label(axA, 'A', x=-0.30, y=1.12)

# ---- B: PDP OHA peak stress ----
axB = fig.add_subplot(gs[0, 1])
axB.plot(pdp_oha['conc'], pdp_oha['preds'][:, 0], '-o', color=C_OHA, lw=1.7, ms=4, mfc='white', mew=1.2)
axB.set_xlabel('Concentration (%)'); axB.set_ylabel('Predicted peak stress (Pa)')
axB.set_title('PDP OHA-GEL (peak)', fontsize=9, fontweight='bold')
set_style(axB); add_label(axB, 'B', x=-0.30, y=1.12)

# ---- C: cycle 1 vs cycle 3 ----
axC = fig.add_subplot(gs[0, 2])
cd = pd.read_csv(os.path.join(DOWN, 'cycle_stability.csv'))
for (mat, conc1), g in cd.groupby(['material', 'conc1']):
    pred = dict(zip(g['cycle'], g['pred_peak_stress']))
    if 1 in pred and 3 in pred:
        color = C_OHA if mat == 'OHA-GEL' else C_ALG
        marker = 'o' if mat == 'OHA-GEL' else 's'
        axC.scatter(pred[1], pred[3], s=55, c=color, marker=marker,
                    edgecolors='white', linewidth=0.6, alpha=0.9, label=f'{mat} {conc1}%')
axC.plot([100, 1200], [100, 1200], 'k--', lw=0.8, alpha=0.45)
axC.set_xlim(100, 1200); axC.set_ylim(100, 1200)
axC.set_xlabel('Cycle 1 peak stress (Pa)'); axC.set_ylabel('Cycle 3 peak stress (Pa)')
axC.legend(fontsize=5.5, loc='center left', bbox_to_anchor=(1.02, 0.5),
           frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
axC.set_title('Cycle stability (1 vs 3)', fontsize=9, fontweight='bold')
set_style(axC); add_label(axC, 'C', x=-0.30, y=1.12)

# ---- D: recon error per material ----
axD = fig.add_subplot(gs[1, 0])
rd = np.load(os.path.join(DOWN, 'recon_error_diagnosis.npz'))
errs = rd['errors']; err_mats = rd['materials']
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
axD.set_title('Recon error by material', fontsize=9, fontweight='bold')
set_style(axD); add_label(axD, 'D', x=-0.30, y=1.12)

# ---- E: mean latent distance bar ----
axE = fig.add_subplot(gs[1, 1])
o = z_tr[latent_labels_tr == 0]; a = z_tr[latent_labels_tr == 1]
d_oo = cdist(o, o).mean(); d_aa = cdist(a, a).mean(); d_oa = cdist(o, a).mean()
mat_lab = ['OHA-OHA', 'Alg-Alg', 'OHA-Alg']
mat_val = [d_oo, d_aa, d_oa]
bars = axE.bar(mat_lab, mat_val, color=[C_OHA, C_ALG, C_ADA], alpha=0.9,
               width=0.55, edgecolor='white', linewidth=0.8)
for b, v in zip(bars, mat_val):
    axE.text(b.get_x()+b.get_width()/2, v+0.05, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
axE.set_ylabel('Mean latent distance')
axE.set_title('Latent distances', fontsize=9, fontweight='bold')
set_style(axE); add_label(axE, 'E', x=-0.30, y=1.12)

# ---- F (NEW): PDP Alginate peak stress ----
axF = fig.add_subplot(gs[1, 2])
axF.plot(pdp_alg['conc'], pdp_alg['preds'][:, 0], '-s', color=C_ALG, lw=1.7, ms=4, mfc='white', mew=1.2)
axF.set_xlabel('Concentration (%)'); axF.set_ylabel('Predicted peak stress (Pa)')
axF.set_title('PDP Alginate (peak)', fontsize=9, fontweight='bold')
set_style(axF); add_label(axF, 'F', x=-0.30, y=1.12)

# ---- G (NEW): cycle 1->2->3 trajectory per formulation ----
axG = fig.add_subplot(gs[2, 0])
for (mat, conc1), g in cd.groupby(['material', 'conc1']):
    g = g.sort_values('cycle')
    color = C_OHA if mat == 'OHA-GEL' else C_ALG
    marker = 'o' if mat == 'OHA-GEL' else 's'
    ls = '-' if mat == 'OHA-GEL' else '--'
    axG.plot(g['cycle'], g['pred_peak_stress'], ls, color=color, marker=marker,
             ms=4, lw=1.5, alpha=0.85, label=f'{mat} {conc1}%')
axG.set_xlabel('Cycle'); axG.set_ylabel('Predicted peak stress (Pa)')
axG.set_xticks([1, 2, 3])
axG.legend(fontsize=5.5, loc='upper right', frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
axG.set_title('Cycle trajectories', fontsize=9, fontweight='bold')
set_style(axG); add_label(axG, 'G', x=-0.30, y=1.12)

# ---- H (NEW): OHA PDP all three indicators (normalised) ----
axH = fig.add_subplot(gs[2, 1])
preds3 = pdp_oha['preds']
inds = ['Peak stress', 'Yield point', 'Toughness']
cols = [C_OURS, C_OHA, C_ALG]
for j in range(3):
    v = preds3[:, j]
    vn = (v - v.min()) / (v.max() - v.min() + 1e-12)
    axH.plot(pdp_oha['conc'], vn, '-o', color=cols[j], lw=1.6, ms=3.5, mfc='white', mew=1.0, label=inds[j])
axH.set_xlabel('Concentration (%)'); axH.set_ylabel('Normalised prediction')
axH.legend(fontsize=6, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.02),
           frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
axH.set_title('OHA PDP (3 indicators)', fontsize=9, fontweight='bold')
set_style(axH); add_label(axH, 'H', x=-0.30, y=1.12)

# ---- I (NEW): latent distance matrix heatmap (3 materials) ----
axI = fig.add_subplot(gs[2, 2])
labels = np.concatenate([latent_labels_tr, np.full(len(latent_test), 2)])
Z = np.vstack([latent_train, latent_test])
names = ['OHA-GEL', 'Alginate', 'ADA-GEL']
Dmat = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        si = Z[labels == i]; sj = Z[labels == j]
        Dmat[i, j] = cdist(si, sj).mean()
im = axI.imshow(Dmat, cmap='YlOrRd', aspect='auto')
for i in range(3):
    for j in range(3):
        axI.text(j, i, f'{Dmat[i,j]:.2f}', ha='center', va='center', fontsize=8.5,
                 fontweight='bold', color='white' if Dmat[i,j] > Dmat.max()*0.6 else '#272727')
axI.set_xticks(range(3)); axI.set_xticklabels(names, fontsize=6.5, rotation=20, ha='right')
axI.set_yticks(range(3)); axI.set_yticklabels(names, fontsize=6.5)
axI.tick_params(length=0)
axI.set_title('Latent distance matrix', fontsize=9, fontweight='bold')
cb = fig.colorbar(im, ax=axI, fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=6)
set_style(axI, grid=False); add_label(axI, 'I', x=-0.30, y=1.12)

plt.tight_layout()
save_fig(fig, 'Fig7')
print('Fig7 v3 saved (3x3, 9 panels)')
