"""SCI-style Fig 5 (2x3 grid): ablation study
Hero: A (peak stress per-sample scatter). B-C: yield/toughness scatter.
D: overall MAE. E: overall R2. F: per-indicator MAE grouped.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sci_common.py'), encoding='utf-8').read())

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(9.6, 5.2))
gs = GridSpec(2, 3, hspace=0.5, wspace=0.4)

models_ab = ['Full', 'No Attn', 'No FiLM']
colors_ab = [C_OURS, BLUE_D, BLUE_M]
markers_ab = ['o', 's', '^']
preds_ab = {'Full': pred_ext, 'No Attn': abl_preds['No Attention'], 'No FiLM': abl_preds['No FiLM']}

# A-C: per-sample scatter
for i in range(3):
    ax = fig.add_subplot(gs[0, i])
    for mname, c, mk in zip(models_ab, colors_ab, markers_ab):
        p = preds_ab[mname]
        ax.scatter(true_ext[:, i], p[:, i], s=50, c=c, marker=mk, edgecolors='white',
                   linewidth=0.6, label=mname, zorder=5, alpha=0.92)
    allv = np.concatenate([true_ext[:, i]] + [preds_ab[m][:, i] for m in models_ab])
    lo, hi = allv.min(), allv.max()
    pad = (hi-lo)*0.12 if hi > lo else 1.0
    lim = [lo-pad, hi+pad]
    ax.plot(lim, lim, 'k--', lw=0.8, alpha=0.45)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f'Measured {targets[i]} ({units[i]})')
    ax.set_ylabel(f'Predicted {targets[i]} ({units[i]})')
    ax.legend(loc='lower right', fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
    ax.text(0.03, 0.97, 'n = 3', transform=ax.transAxes, fontsize=7, va='top', color=NEU_D, style='italic')
    set_style(ax)
    add_label(ax, chr(65+i), x=-0.14, y=1.06)
    if i == 0:
        ax.set_title('Hero: peak stress', fontsize=9, fontweight='bold')

# D: overall MAE
ax = fig.add_subplot(gs[1, 0])
mae_vals = [mae['Ours']['overall'], mae['No Attn']['overall'], mae['No FiLM']['overall']]
bars = ax.bar(models_ab, mae_vals, color=colors_ab, alpha=0.9, width=0.6, edgecolor='white', linewidth=0.8)
for b, v in zip(bars, mae_vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.8, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
ax.set_ylabel('Overall MAE')
ax.set_ylim(0, max(mae_vals)*1.3)
set_style(ax); add_label(ax, 'D', x=-0.14, y=1.06)

# E: overall R2
ax = fig.add_subplot(gs[1, 1])
r2_vals = [r2_ext['Ours'], r2_ext['No Attn'], r2_ext['No FiLM']]
bars = ax.bar(models_ab, r2_vals, color=colors_ab, alpha=0.9, width=0.6, edgecolor='white', linewidth=0.8)
for b, v in zip(bars, r2_vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
ax.set_ylabel('Overall $R^2$')
ax.set_ylim(0, 1.05)
set_style(ax); add_label(ax, 'E', x=-0.14, y=1.06)

# F: per-indicator MAE grouped
ax = fig.add_subplot(gs[1, 2])
x = np.arange(3)
w = 0.26
mae_map = {'Full': 'Ours', 'No Attn': 'No Attn', 'No FiLM': 'No FiLM'}
for j, (mn, mc) in enumerate(zip(models_ab, colors_ab)):
    mk = mae_map[mn]
    vals = [mae[mk]['peak'], mae[mk]['yield']*20, mae[mk]['tough']]
    ax.bar(x + (j-1)*w, vals, width=w, color=mc, alpha=0.85, edgecolor='white', linewidth=0.5, label=mn)
ax.set_xticks(x); ax.set_xticklabels(['Peak', 'Yield×20', 'Tough'])
ax.set_ylabel('MAE (scaled)')
ax.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(ax); add_label(ax, 'F', x=-0.14, y=1.06)

plt.tight_layout()
save_fig(fig, 'Fig5')
