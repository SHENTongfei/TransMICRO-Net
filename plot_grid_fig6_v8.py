"""SCI Fig 6 v8 - Final fixes (C panel layout + D ylabel scale)
- A: Hero attention profile (×10⁻⁷)
- B: per-sample traces (×10⁻⁷)
- C: GROUPED bar of ABSOLUTE |ΔMSE| per segment × per target (legend OUTSIDE-LEFT)
- D: Variance (×10⁻⁸, was 1e11 wrong)
- E: Sensitivity heatmap
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_common.py'), encoding='utf-8').read())

from matplotlib.gridspec import GridSpec
import numpy as np

DOWN = os.path.join(ANALYSIS_DIR, 'downstream')
attn_train = np.load(os.path.join(DOWN, 'attn_weights_train.npy'))
attn_labels = np.load(os.path.join(DOWN, 'attn_labels_train.npy'))
perm_imp = np.load(os.path.join(DOWN, 'permutation_importance.npy'))
perm_seg = np.load(os.path.join(DOWN, 'permutation_segments.npy'))

fig = plt.figure(figsize=(12.0, 6.4))
gs = GridSpec(2, 3, width_ratios=[1.3, 1.0, 1.0], hspace=0.75, wspace=0.6)

# A (HERO)
axA = fig.add_subplot(gs[:, 0])
for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
    mask = attn_labels == mat
    if mask.sum() == 0: continue
    dev = (attn_train[mask] - 0.005) * 1e6  # put on 10⁻⁷ scale
    axA.plot(strain, dev.mean(axis=0), color=color, lw=1.8, label=name)
    axA.fill_between(strain, dev.mean(axis=0)-dev.std(axis=0),
                    dev.mean(axis=0)+dev.std(axis=0),
                    color=color, alpha=0.15)
axA.axhline(0, color=NEU_D, ls='--', lw=0.7, alpha=0.6)
axA.set_xlabel('Strain')
axA.set_ylabel('Attention deviation (×$10^{-7}$)')
axA.legend(fontsize=7, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none', loc='upper right')
set_style(axA)
add_label(axA, 'A', x=-0.18, y=1.05)
axA.set_title('Hero: attention profile', fontsize=9, fontweight='bold')

# B — per-sample traces
axB = fig.add_subplot(gs[0, 1])
for i in range(len(attn_train)):
    color = C_OHA if attn_labels[i] == 0 else C_ALG
    axB.plot(strain, (attn_train[i]-0.005)*1e6, color=color, lw=0.5, alpha=0.35)
axB.axhline(0, color=NEU_D, ls='--', lw=0.7, alpha=0.6)
axB.set_xlabel('Strain'); axB.set_ylabel('Attention dev. (×$10^{-7}$)')
set_style(axB); add_label(axB, 'B', x=-0.14, y=1.06)

# C: GROUPED bar of ABSOLUTE |ΔMSE| per segment × per target
axC = fig.add_subplot(gs[1, 1])
segs = perm_seg[:8]
imps = np.abs(perm_imp[:8, :])  # absolute values
n_targets = imps.shape[1]
x = np.arange(len(segs))
w = 0.26
colors_targets = [C_OURS, C_OHA, C_ALG]
for j, target in enumerate(targets):
    axC.bar(x + (j-1)*w, imps[:, j], width=w, color=colors_targets[j],
           edgecolor='white', linewidth=0.7)
# Annotate only the S0 Peak value (the dominant signal)
v_peak_s0 = imps[0, 0]
axC.text(x[0] - w, v_peak_s0 * 1.5, f'{v_peak_s0:.0f}', ha='center', fontsize=7,
        fontweight='bold', color=C_OURS)
axC.set_xticks(x)
axC.set_xticklabels([f'S{i}' for i in range(len(segs))], fontsize=8)
axC.set_xlabel('Strain segment (200-pt curve split into 8)')
axC.set_ylabel('Absolute $|\\Delta MSE|$ per segment')
axC.set_yscale('symlog', linthresh=1)
# Legend OUTSIDE-LEFT (no overlap with bars or title)
axC.legend(labels=['Peak stress', 'Yield point', 'Toughness'],
          handles=[plt.Rectangle((0,0),1,1, color=C_OURS, ec='white'),
                   plt.Rectangle((0,0),1,1, color=C_OHA, ec='white'),
                   plt.Rectangle((0,0),1,1, color=C_ALG, ec='white')],
          fontsize=6.5, loc='upper left', bbox_to_anchor=(-0.02, -0.18),
          ncol=3, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axC); add_label(axC, 'C', x=-0.18, y=1.06)
axC.set_title('S0 dominates permutation $|\\Delta MSE|$', fontsize=9, fontweight='bold', pad=12)

# D — variance (FIXED: actual scale is 10⁻⁸, was 1e11 wrong in v7)
axD = fig.add_subplot(gs[0, 2])
var_by_mat = [attn_train[attn_labels == m].var(axis=0) for m in [0, 1]]
for v, l, c in zip(var_by_mat, ['OHA-GEL', 'Alginate'], [C_OHA, C_ALG]):
    axD.plot(strain, v, color=c, lw=1.5, label=l)  # raw values, no scaling
axD.set_xlabel('Strain'); axD.set_ylabel('Variance of attention')  # no scaling factor
axD.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axD); add_label(axD, 'D', x=-0.14, y=1.06)

# E — sensitivity heatmap
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
print('Fig6 v8 saved')