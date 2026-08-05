"""SCI-style Fig 3 v2 (2x3 grid with hero panel): reconstruction performance
Upgrades vs v1:
  D: horizontal broken-down bar showing each physics component value + total
  E: slope chart (dumbbell) showing TransMICRO vs RF gap per indicator
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_common.py'), encoding='utf-8').read())

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(9.6, 5.8))
gs = GridSpec(2, 3, width_ratios=[1.25, 1, 1], height_ratios=[1, 1], hspace=0.50, wspace=0.40)

# ---- HERO panel A: example reconstruction (spans left column, tall) ----
axA = fig.add_subplot(gs[:, 0])
d = np.load(os.path.join(MODEL_DIR, 'fold1_val_predictions.npz'))
rt, rp = d['recon_true'], d['recon_pred']
i = 0
true_c = smooth_curve(rt[i])
pred_c = smooth_curve(rp[i])
axA.plot(strain, true_c, '-', color='#272727', lw=1.8, label='Measured')
axA.plot(strain, pred_c, '--', color=C_OURS, lw=1.7, label='Reconstructed')
axA.fill_between(strain, true_c, pred_c, alpha=0.12, color=C_OURS)
axA.set_xlabel('Strain')
axA.set_ylabel('Normalised stress')
axA.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axA)
add_label(axA, 'A', x=-0.14, y=1.03)
axA.set_title('Example curve reconstruction', fontsize=9, fontweight='bold')

# ---- B: per-fold reconstruction R2 ----
axB = fig.add_subplot(gs[0, 1])
folds = [1, 2, 3]
r2s = [fold_recon[f]['r2_mean'] for f in folds]
bars = axB.bar([f'Fold {f}' for f in folds], r2s, color=[BLUE_D, BLUE_M, BLUE_S],
               width=0.55, edgecolor='white', linewidth=0.8)
for b, v in zip(bars, r2s):
    axB.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
axB.axhline(0.7, color=NEU_D, ls='--', lw=0.8, alpha=0.6, label='threshold 0.7')
axB.set_ylabel('Recon. $R^2$')
axB.set_ylim(0, 1.05)
axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axB.legend(loc='lower right', fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axB); add_label(axB, 'B', x=-0.3, y=1.15)

# ---- C: per-fold reconstruction RMSE ----
axC = fig.add_subplot(gs[0, 2])
rmses = [fold_recon[f]['rmse'] for f in folds]
bars = axC.bar([f'Fold {f}' for f in folds], rmses, color=[BLUE_D, BLUE_M, BLUE_S],
               width=0.55, edgecolor='white', linewidth=0.8)
for b, v in zip(bars, rmses):
    axC.text(b.get_x()+b.get_width()/2, v+0.006, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
axC.set_ylabel('Recon. RMSE')
axC.legend(loc='upper right', fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axC); add_label(axC, 'C', x=-0.3, y=1.15)

# ---- D: HORIZONTAL broken-down bar (each component value + total label) ----
axD = fig.add_subplot(gs[1, 1])
mono_est = [p * 0.585 for p in physics_folds]
cons_est = [p - m for p, m in zip(physics_folds, mono_est)]
y_pos = np.arange(len(folds))
bar_h = 0.55
# two stacked segments per fold
for k, (m, c) in enumerate(zip(mono_est, cons_est)):
    # mono segment (BLUE_D)
    axD.barh(y_pos[k], m, height=bar_h, color=BLUE_D, edgecolor='white', linewidth=0.7)
    # consistency segment (TEAL), stacked to the right
    axD.barh(y_pos[k], c, height=bar_h, left=m, color=TEAL, edgecolor='white', linewidth=0.7)
    # annotate each segment with value (inside)
    axD.text(m/2, y_pos[k], f'{m:.1f}', ha='center', va='center',
             fontsize=7.5, fontweight='bold', color='white' if m > 1.5 else NEU_D)
    axD.text(m + c/2, y_pos[k], f'{c:.1f}', ha='center', va='center',
             fontsize=7.5, fontweight='bold', color='white' if c > 1.5 else NEU_D)
    # total at right of bar
    total = m + c
    axD.text(total + 0.18, y_pos[k], f'= {total:.1f}', va='center', fontsize=8,
             fontweight='bold', color=NEU_D)
axD.axvline(9, color=NEU_D, ls='--', lw=0.8, alpha=0.5)
axD.set_yticks(y_pos); axD.set_yticklabels([f'Fold {f}' for f in folds])
axD.set_xlabel('Physics score')
axD.set_xlim(0, 13.0)
axD.invert_yaxis()  # Fold 1 on top
# legend (placed BELOW axes to avoid covering bars)
axD.barh([], [], color=BLUE_D, label='Monotonicity')
axD.barh([], [], color=TEAL, label='Consistency')
axD.legend(loc='upper left', fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none',
           bbox_to_anchor=(1.02, 1.0))
set_style(axD); add_label(axD, 'D', x=-0.3, y=1.15)

# ---- E: SLOPE CHART (dumbbell) showing TransMICRO vs RF gap per indicator ----
axE = fig.add_subplot(gs[1, 2])
cv_r2 = {}
for _, row in summary.iterrows():
    if row['Model'] in ('TransMICRO-Net (Ours)', 'Random Forest (5 trees)'):
        cv_r2[row['Model']] = [row['peak_stress_R2_CV'], row['yield_point_R2_CV'], row['toughness_R2_CV']]
labels = ['Peak\nstress', 'Yield\npoint', 'Toughness']
y_pos = np.arange(len(labels))
ours_vals = cv_r2['TransMICRO-Net (Ours)']
rf_vals = cv_r2['Random Forest (5 trees)']
# connecting lines + endpoints
for k in range(len(labels)):
    # line
    line_color = UP_GREEN if ours_vals[k] > rf_vals[k] else DOWN_RED
    axE.plot([rf_vals[k], ours_vals[k]], [y_pos[k], y_pos[k]],
             color=line_color, lw=1.6, alpha=0.7, zorder=2)
    # RF dot (blue square)
    axE.scatter([rf_vals[k]], [y_pos[k]], s=120, c=BLUE_D, marker='s',
                edgecolors='white', linewidth=1.0, zorder=4, label='RF' if k == 0 else None)
    # Ours dot (red circle)
    axE.scatter([ours_vals[k]], [y_pos[k]], s=140, c=C_OURS, marker='o',
                edgecolors='white', linewidth=1.0, zorder=5, label='TransMICRO' if k == 0 else None)
    # delta annotation
    delta = ours_vals[k] - rf_vals[k]
    arrow = '↑' if delta > 0 else '↓'
    axE.text(max(ours_vals[k], rf_vals[k]) + 0.03, y_pos[k],
             f'{arrow}{delta:+.2f}', va='center', fontsize=7.5,
             fontweight='bold', color=line_color)
axE.axvline(0.7, color=NEU_D, ls='--', lw=0.6, alpha=0.5)
axE.axvline(0, color=NEU_D, lw=0.6)
axE.set_yticks(y_pos); axE.set_yticklabels(labels)
axE.set_xlabel('CV $R^2$')
axE.set_xlim(-0.05, 1.35)
axE.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
# legend placed outside right to avoid covering Peak stress row at top
axE.legend(loc='upper left', fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none',
           bbox_to_anchor=(1.02, 1.0))
set_style(axE); add_label(axE, 'E', x=-0.3, y=1.15)

plt.tight_layout()
save_fig(fig, 'Fig3')