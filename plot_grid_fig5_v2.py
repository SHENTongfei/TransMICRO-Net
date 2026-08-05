"""SCI-style Fig 5 v2 (2x3 grid): ablation study
Upgrades vs v1:
  F: replaces "Yieldx20" hack with model-mean normalised MAE per indicator
     (all on [0, 1.5] scale, comparable across indicators)
     + adds R² line overlay on secondary y-axis
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_common.py'), encoding='utf-8').read())

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(9.6, 5.4))
gs = GridSpec(2, 3, hspace=0.55, wspace=0.42)

models_ab = ['Full', 'No Attn', 'No FiLM']
colors_ab = [C_OURS, BLUE_D, BLUE_M]
markers_ab = ['o', 's', '^']
preds_ab = {'Full': pred_ext, 'No Attn': abl_preds['No Attention'], 'No FiLM': abl_preds['No FiLM']}

# A-C: per-sample scatter with diagonal and 1:1 region
for i in range(3):
    ax = fig.add_subplot(gs[0, i])
    for mname, c, mk in zip(models_ab, colors_ab, markers_ab):
        p = preds_ab[mname]
        ax.scatter(true_ext[:, i], p[:, i], s=55, c=c, marker=mk, edgecolors='white',
                   linewidth=0.6, label=mname, zorder=5, alpha=0.92)
    allv = np.concatenate([true_ext[:, i]] + [preds_ab[m][:, i] for m in models_ab])
    lo, hi = allv.min(), allv.max()
    pad = (hi-lo)*0.12 if hi > lo else 1.0
    lim = [lo-pad, hi+pad]
    ax.plot(lim, lim, 'k--', lw=0.8, alpha=0.45)
    # +-- 5% band
    band = 0.05 * (hi - lo)
    ax.fill_between(lim, [lim[0]-band]*2, [lim[0]+band]*2, alpha=0.0)  # placeholder
    # 5% tolerance band around diagonal
    diag_lo = [l - (hi-lo)*0.05 for l in lim]
    diag_hi = [l + (hi-lo)*0.05 for l in lim]
    ax.fill_between(lim, diag_lo, diag_hi, color=NEU_L, alpha=0.35, zorder=1,
                    label='±5% band')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f'Measured {targets[i]} ({units[i]})')
    ax.set_ylabel(f'Predicted {targets[i]} ({units[i]})')
    ax.legend(loc='lower right', fontsize=5.5, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
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

# F: NORMALISED per-indicator MAE grouped bar (no x20 hack) + R² overlay line
ax = fig.add_subplot(gs[1, 2])
# Normalise each indicator's MAE by its model-mean so all on comparable [0, 1.5] scale
indicators = ['peak', 'yield', 'tough']
ind_labels = ['Peak stress', 'Yield point', 'Toughness']
# mae dict uses 'Ours' not 'Full' (plot_common.py source of truth)
model_key_map = {'Full': 'Ours', 'No Attn': 'No Attn', 'No FiLM': 'No FiLM'}
mae_per_model_ind = {m: [mae[model_key_map[m]]['peak'],
                          mae[model_key_map[m]]['yield'],
                          mae[model_key_map[m]]['tough']] for m in models_ab}
# model-mean per indicator (across models)
ind_means = np.mean([[mae_per_model_ind[m][k] for m in models_ab] for k in range(3)], axis=1)
norm_mae = {m: [mae_per_model_ind[m][k] / ind_means[k] for k in range(3)] for m in models_ab}
x = np.arange(3)
w = 0.26
for j, (mn, mc) in enumerate(zip(models_ab, colors_ab)):
    ax.bar(x + (j-1)*w, norm_mae[mn], width=w, color=mc, alpha=0.85,
           edgecolor='white', linewidth=0.5, label=mn)
# dashed line at y=1.0 = model mean baseline
ax.axhline(1.0, color=NEU_D, ls='--', lw=0.8, alpha=0.6, label='model mean')
# value annotation per bar
for j, mn in enumerate(models_ab):
    for k in range(3):
        v = norm_mae[mn][k]
        ax.text(x[k] + (j-1)*w, v + 0.03, f'{v:.2f}', ha='center', fontsize=6.5,
                fontweight='bold', color=NEU_D)
ax.set_xticks(x); ax.set_xticklabels(ind_labels, fontsize=7.5)
ax.set_ylabel('Normalised MAE\n(model-mean=1)')
ax.set_ylim(0, 1.7)
ax.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none',
          loc='upper left')
set_style(ax); add_label(ax, 'F', x=-0.14, y=1.06)

plt.tight_layout()
save_fig(fig, 'Fig5')