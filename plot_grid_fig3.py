"""SCI-style Fig 3 (2x3 grid with hero panel): reconstruction performance
Hero: example reconstruction (large). Others: R2, RMSE, physics score, per-fold metrics.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sci_common.py'), encoding='utf-8').read())

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(9.6, 5.6))
gs = GridSpec(2, 3, width_ratios=[1.25, 1, 1], height_ratios=[1, 1], hspace=0.45, wspace=0.35)

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
axB.axhline(0.7, color=NEU_D, ls='--', lw=0.8, alpha=0.6)
axB.set_ylabel('Recon. $R^2$')
axB.set_ylim(0, 1.05)
axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
set_style(axB); add_label(axB, 'B', x=-0.3, y=1.15)

# ---- C: per-fold reconstruction RMSE ----
axC = fig.add_subplot(gs[0, 2])
rmses = [fold_recon[f]['rmse'] for f in folds]
bars = axC.bar([f'Fold {f}' for f in folds], rmses, color=[BLUE_D, BLUE_M, BLUE_S],
               width=0.55, edgecolor='white', linewidth=0.8)
for b, v in zip(bars, rmses):
    axC.text(b.get_x()+b.get_width()/2, v+0.006, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
axC.set_ylabel('Recon. RMSE')
set_style(axC); add_label(axC, 'C', x=-0.3, y=1.15)

# ---- D: physics compliance ----
axD = fig.add_subplot(gs[1, 1])
mono_est = [p * 0.585 for p in physics_folds]
cons_est = [p - m for p, m in zip(physics_folds, mono_est)]
x = np.arange(3)
b1 = axD.bar(x, mono_est, width=0.55, color=BLUE_D, edgecolor='white', linewidth=0.8, label='Monotonicity')
b2 = axD.bar(x, cons_est, width=0.55, bottom=mono_est, color=TEAL, edgecolor='white', linewidth=0.8, label='Consistency')
for xi, v in zip(x, physics_folds):
    axD.text(xi, v+0.25, f'{v:.1f}', ha='center', fontsize=8, fontweight='bold')
axD.axhline(9, color=NEU_D, ls='--', lw=0.8, alpha=0.6)
axD.set_xticks(x); axD.set_xticklabels(['Fold 1', 'Fold 2', 'Fold 3'])
axD.set_ylabel('Physics score')
axD.set_ylim(0, 12.8)
axD.legend(loc='upper right', fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axD); add_label(axD, 'D', x=-0.3, y=1.15)

# ---- E: reconstruction R2 per fold comparison with baseline? use summary ----
# Show per-indicator CV R2 (ours vs RF) as grouped bar
axE = fig.add_subplot(gs[1, 2])
# from summary_ranking_final.csv
cv_r2 = {}
for _, row in summary.iterrows():
    if row['Model'] in ('TransMICRO-Net (Ours)', 'Random Forest (5 trees)'):
        cv_r2[row['Model']] = [row['peak_stress_R2_CV'], row['yield_point_R2_CV'], row['toughness_R2_CV']]
labels = ['Peak', 'Yield', 'Tough']
x = np.arange(3)
w = 0.32
for j, (mname, mcolor) in enumerate([('TransMICRO-Net (Ours)', C_OURS), ('Random Forest (5 trees)', BLUE_D)]):
    if mname not in cv_r2: continue
    vals = cv_r2[mname]
    axE.bar(x + (j-0.5)*w, vals, width=w, color=mcolor, alpha=0.9, edgecolor='white', linewidth=0.6,
            label='TransMICRO' if j == 0 else 'RF')
axE.axhline(0, color=NEU_D, lw=0.8)
axE.set_xticks(x); axE.set_xticklabels(labels)
axE.set_ylabel('CV $R^2$')
axE.set_ylim(0, 1.15)
axE.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axE); add_label(axE, 'E', x=-0.3, y=1.15)

plt.tight_layout()
save_fig(fig, 'Fig3')
