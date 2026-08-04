"""SCI-style Fig 6 (2x3 grid with hero): attention + permutation
Hero: attention profiles (A spans 2 rows). B: per-sample. C/D: permutation. E: heatmap.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sci_common.py'), encoding='utf-8').read())

from matplotlib.gridspec import GridSpec

DOWN = os.path.join(ANALYSIS_DIR, 'downstream')
attn_train = np.load(os.path.join(DOWN, 'attn_weights_train.npy'))
attn_labels = np.load(os.path.join(DOWN, 'attn_labels_train.npy'))
perm_imp = np.load(os.path.join(DOWN, 'permutation_importance.npy'))
perm_seg = np.load(os.path.join(DOWN, 'permutation_segments.npy'))

fig = plt.figure(figsize=(10.0, 5.4))
gs = GridSpec(2, 3, width_ratios=[1.4, 1, 1], hspace=0.5, wspace=0.42)

# HERO A: attention mean +/- std (spans left column, tall)
axA = fig.add_subplot(gs[:, 0])
for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
    mask = attn_labels == mat
    if mask.sum() == 0: continue
    dev = (attn_train[mask] - 0.005) * 1e3
    axA.plot(strain, dev.mean(axis=0), color=color, lw=1.8, label=name)
    axA.fill_between(strain, dev.mean(axis=0)-dev.std(axis=0), dev.mean(axis=0)+dev.std(axis=0),
                    color=color, alpha=0.18)
axA.axhline(0, color=NEU_D, ls='--', lw=0.7, alpha=0.6)
axA.set_xlabel('Strain')
axA.set_ylabel('Attention deviation (×$10^{-3}$)')
axA.legend(fontsize=7, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axA)
add_label(axA, 'A', x=-0.18, y=1.05)
axA.set_title('Hero: attention profile', fontsize=9, fontweight='bold')

# B: per-sample traces (faint, right column top)
axB = fig.add_subplot(gs[0, 1])
for i in range(len(attn_train)):
    color = C_OHA if attn_labels[i] == 0 else C_ALG
    axB.plot(strain, (attn_train[i]-0.005)*1e3, color=color, lw=0.5, alpha=0.35)
axB.axhline(0, color=NEU_D, ls='--', lw=0.7, alpha=0.6)
axB.set_xlabel('Strain'); axB.set_ylabel('Attention dev. (×$10^{-3}$)')
set_style(axB); add_label(axB, 'B', x=-0.14, y=1.06)

# C: permutation importance (log-scale)
axC = fig.add_subplot(gs[0, 2])
segs = perm_seg[:8]
imps = perm_imp[:8, :]
x = np.arange(len(segs))
w = 0.26
for j, (tname, tcolor) in enumerate(zip(targets, [C_OURS, C_OHA, C_ALG])):
    axC.bar(x + (j-1)*w, imps[:, j], width=w, color=tcolor, alpha=0.9,
           edgecolor='white', linewidth=0.5, label=tname)
axC.set_yscale('symlog', linthresh=10)
axC.set_xticks(x)
axC.set_xticklabels([f'S{i}' for i in range(len(segs))], fontsize=6, rotation=45)
axC.set_ylabel('ΔMSE (symlog)')
axC.legend(fontsize=6, ncol=3, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axC); add_label(axC, 'C', x=-0.14, y=1.06)

# D: attention variance (right column bottom)
axD = fig.add_subplot(gs[1, 1])
var_by_mat = [attn_train[attn_labels == m].var(axis=0) for m in [0, 1]]
labels_m = ['OHA-GEL', 'Alginate']
colors_m = [C_OHA, C_ALG]
for v, l, c in zip(var_by_mat, labels_m, colors_m):
    axD.plot(strain, v*1e8, color=c, lw=1.5, label=l)
axD.set_xlabel('Strain'); axD.set_ylabel('Variance (×$10^{-8}$)')
axD.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axD); add_label(axD, 'D', x=-0.14, y=1.06)

# E: gradient sensitivity (heatmap)
gs_sens = np.load(os.path.join(DOWN, 'gradient_sensitivity.npz'))
sens = gs_sens['sensitivity']
axE = fig.add_subplot(gs[1, 2])
vmax = np.percentile(np.abs(sens), 95)
im = axE.imshow(sens, aspect='auto', cmap='Reds', vmin=0, vmax=vmax, interpolation='nearest')
axE.set_xlabel('Strain index'); axE.set_ylabel('Sample')
cb = fig.colorbar(im, ax=axE, fraction=0.046, pad=0.02)
cb.ax.tick_params(labelsize=6)
set_style(axE, grid=False); add_label(axE, 'E', x=-0.14, y=1.06)

plt.tight_layout()
save_fig(fig, 'Fig6')