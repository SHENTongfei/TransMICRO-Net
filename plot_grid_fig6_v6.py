"""SCI Fig 6 v6 - Simplified C panel (no overlapping numeric labels)
- A: Hero attention profile
- B: per-sample traces
- C: 100% STACKED contribution (legend BELOW, no numeric labels, key takeaway in title)
- D: Variance
- E: Sensitivity heatmap
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sci_common.py'), encoding='utf-8').read())

from matplotlib.gridspec import GridSpec
import numpy as np

DOWN = os.path.join(ANALYSIS_DIR, 'downstream')
attn_train = np.load(os.path.join(DOWN, 'attn_weights_train.npy'))
attn_labels = np.load(os.path.join(DOWN, 'attn_labels_train.npy'))
perm_imp = np.load(os.path.join(DOWN, 'permutation_importance.npy'))
perm_seg = np.load(os.path.join(DOWN, 'permutation_segments.npy'))

fig = plt.figure(figsize=(11.5, 6.2))
gs = GridSpec(2, 3, width_ratios=[1.3, 1.0, 1.0], hspace=0.6, wspace=0.55)

# A (HERO)
axA = fig.add_subplot(gs[:, 0])
for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
    mask = attn_labels == mat
    if mask.sum() == 0: continue
    dev = (attn_train[mask] - 0.005) * 1e3
    axA.plot(strain, dev.mean(axis=0), color=color, lw=1.8, label=name)
    axA.fill_between(strain, dev.mean(axis=0)-dev.std(axis=0),
                    dev.mean(axis+0)-dev.std(axis=0) if False else dev.mean(axis=0)+dev.std(axis=0),
                    color=color, alpha=0.15)
axA.axhline(0, color=NEU_D, ls='--', lw=0.7, alpha=0.6)
axA.set_xlabel('Strain')
axA.set_ylabel('Attention deviation (×$10^{-3}$)')
axA.legend(fontsize=7, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none', loc='upper right')
set_style(axA)
add_label(axA, 'A', x=-0.18, y=1.05)
axA.set_title('Hero: attention profile', fontsize=9, fontweight='bold')

# B
axB = fig.add_subplot(gs[0, 1])
for i in range(len(attn_train)):
    color = C_OHA if attn_labels[i] == 0 else C_ALG
    axB.plot(strain, (attn_train[i]-0.005)*1e3, color=color, lw=0.5, alpha=0.35)
axB.axhline(0, color=NEU_D, ls='--', lw=0.7, alpha=0.6)
axB.set_xlabel('Strain'); axB.set_ylabel('Attention dev. (×$10^{-3}$)')
set_style(axB); add_label(axB, 'B', x=-0.14, y=1.06)

# C: 100% stacked bar (NO numeric labels in/near bars)
axC = fig.add_subplot(gs[1, 1])
segs = perm_seg[:8]
imps = perm_imp[:8, :]
total_per_target = imps.sum(axis=0, keepdims=True)
perc = (imps / total_per_target) * 100
x = np.arange(len(segs))
bottom = np.zeros(len(segs))
colors_stack = [C_OURS, C_OHA, C_ALG]
for j, target in enumerate(targets):
    axC.bar(x, perc[:, j], width=0.7, bottom=bottom, color=colors_stack[j],
           edgecolor='white', linewidth=0.7, label=target)
    bottom += perc[:, j]
# Annotate only the dominant component of each segment (Peak for S0, Yield for S1..S7)
for k in range(len(segs)):
    dom_j = int(np.argmax(perc[k, :]))
    dom_v = perc[k, dom_j]
    bot = sum(float(perc[k, jj]) for jj in range(dom_j))
    y_pos = bot + dom_v / 2
    target_short = targets[dom_j].replace('Peak stress', 'Peak').replace('Yield point', 'Yield').replace('Toughness', 'Tough.')
    color = 'white' if dom_v >= 25 else NEU_D
    axC.text(k, y_pos, f'{target_short}\n{dom_v:.0f}%', ha='center', va='center',
            fontsize=6.5, fontweight='bold', color=color)
axC.set_xticks(x)
axC.set_xticklabels([f'S{i}' for i in range(len(segs))], fontsize=7)
axC.set_xlabel('Strain segment (200-pt curve split into 8)')
axC.set_ylabel('Share of total ΔMSE (%)')
axC.set_ylim(0, 108)
# Legend BELOW the panel (no overlap with bars)
axC.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.30),
          ncol=3, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axC); add_label(axC, 'C', x=-0.18, y=1.06)
axC.set_title('Stacked ΔMSE share per segment (S0 dominates)', fontsize=9, fontweight='bold')

# D
axD = fig.add_subplot(gs[0, 2])
var_by_mat = [attn_train[attn_labels == m].var(axis=0) for m in [0, 1]]
for v, l, c in zip(var_by_mat, ['OHA-GEL', 'Alginate'], [C_OHA, C_ALG]):
    axD.plot(strain, v*1e8, color=c, lw=1.5, label=l)
axD.set_xlabel('Strain'); axD.set_ylabel('Variance (×$10^{-8}$)')
axD.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axD); add_label(axD, 'D', x=-0.14, y=1.06)

# E
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
print('Fig6 v6 saved')