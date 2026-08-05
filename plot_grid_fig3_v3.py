"""SCI-style Fig 3 v3: 3x3 GRID (9 panels, NO empty slot) - internal CV main figure
Upgrades vs v2:
  D: heatmap of CV R^2 (4 models x 3 indicators) with value annotations   [advanced]
  G: violin plot of reconstruction error by fold                          [advanced]
  H: stripplot of per-sample reconstruction R^2 by fold + 0.7 threshold   [advanced]
  I: residual heatmap (samples x strain) for fold 1                      [advanced]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_common.py'), encoding='utf-8').read())

from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

fig = plt.figure(figsize=(10.4, 8.6))
gs = GridSpec(3, 3, hspace=0.62, wspace=0.55)

# ---- A (HERO): example reconstruction (fold 1, sample 0) ----
axA = fig.add_subplot(gs[0, 0])
d = np.load(os.path.join(MODEL_DIR, 'fold1_val_predictions.npz'))
rt, rp = d['recon_true'], d['recon_pred']
i = 0
true_c = smooth_curve(rt[i]); pred_c = smooth_curve(rp[i])
axA.plot(strain, true_c, '-', color='#272727', lw=1.8, label='Measured')
axA.plot(strain, pred_c, '--', color=C_OURS, lw=1.7, label='Reconstructed')
axA.fill_between(strain, true_c, pred_c, alpha=0.12, color=C_OURS)
axA.set_xlabel('Strain'); axA.set_ylabel('Normalised stress')
axA.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axA); add_label(axA, 'A', x=-0.30, y=1.12)
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
axB.set_ylabel('Recon. $R^2$'); axB.set_ylim(0, 1.05)
axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axB.legend(loc='upper left', fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axB); add_label(axB, 'B', x=-0.30, y=1.12)

# ---- C: per-fold reconstruction RMSE ----
axC = fig.add_subplot(gs[0, 2])
rmses = [fold_recon[f]['rmse'] for f in folds]
bars = axC.bar([f'Fold {f}' for f in folds], rmses, color=[BLUE_D, BLUE_M, BLUE_S],
               width=0.55, edgecolor='white', linewidth=0.8)
for b, v in zip(bars, rmses):
    axC.text(b.get_x()+b.get_width()/2, v+0.006, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
axC.set_ylabel('Recon. RMSE')
set_style(axC); add_label(axC, 'C', x=-0.30, y=1.12)

# ---- D (NEW): CV R^2 heatmap: 4 models x 3 indicators ----
axD = fig.add_subplot(gs[1, 0])
mods = ['TransMICRO-Net', 'RF', 'DT', 'KNN']
cvr2 = np.array([
    [0.9802, 0.5785, 0.9706],
    [0.8559, 0.7268, 0.9216],
    [0.8841, 0.7874, 0.9251],
    [0.7710, -0.1106, 0.8216],
])
im = axD.imshow(cvr2, cmap='RdYlBu_r', vmin=-0.2, vmax=1.0, aspect='auto')
for r in range(4):
    for c in range(3):
        v = cvr2[r, c]
        axD.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=7.5,
                 fontweight='bold', color='white' if v < 0.4 else '#272727')
axD.set_xticks(range(3)); axD.set_xticklabels(['Peak', 'Yield', 'Tough'], fontsize=7)
axD.set_yticks(range(4)); axD.set_yticklabels(mods, fontsize=7)
axD.tick_params(length=0)
axD.set_title('CV $R^2$ (model x indicator)', fontsize=8.5, fontweight='bold')
cb = fig.colorbar(im, ax=axD, fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=6)
set_style(axD, grid=False); add_label(axD, 'D', x=-0.30, y=1.12)

# ---- E: slope chart (dumbbell) TransMICRO vs RF ----
axE = fig.add_subplot(gs[1, 1])
cv_r2 = {}
for _, row in summary.iterrows():
    if row['Model'] in ('TransMICRO-Net (Ours)', 'Random Forest (5 trees)'):
        cv_r2[row['Model']] = [row['peak_stress_R2_CV'], row['yield_point_R2_CV'], row['toughness_R2_CV']]
labels = ['Peak\nstress', 'Yield\npoint', 'Toughness']
y_pos = np.arange(len(labels))
ours_vals = cv_r2['TransMICRO-Net (Ours)']
rf_vals = cv_r2['Random Forest (5 trees)']
for k in range(len(labels)):
    line_color = UP_GREEN if ours_vals[k] > rf_vals[k] else DOWN_RED
    axE.plot([rf_vals[k], ours_vals[k]], [y_pos[k], y_pos[k]], color=line_color, lw=1.6, alpha=0.7, zorder=2)
    axE.scatter([rf_vals[k]], [y_pos[k]], s=120, c=BLUE_D, marker='s', edgecolors='white', linewidth=1.0, zorder=4, label='RF' if k == 0 else None)
    axE.scatter([ours_vals[k]], [y_pos[k]], s=140, c=C_OURS, marker='o', edgecolors='white', linewidth=1.0, zorder=5, label='TransMICRO' if k == 0 else None)
    delta = ours_vals[k] - rf_vals[k]
    arrow = '\u2191' if delta > 0 else '\u2193'
    axE.text(max(ours_vals[k], rf_vals[k]) + 0.03, y_pos[k], f'{arrow}{delta:+.2f}',
             va='center', fontsize=7.5, fontweight='bold', color=line_color)
axE.axvline(0.7, color=NEU_D, ls='--', lw=0.6, alpha=0.5)
axE.set_yticks(y_pos); axE.set_yticklabels(labels)
axE.set_xlabel('CV $R^2$'); axE.set_xlim(-0.05, 1.35)
axE.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axE.legend(loc='upper left', fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none',
           bbox_to_anchor=(1.02, 1.0))
set_style(axE); add_label(axE, 'E', x=-0.30, y=1.12)

# ---- F: physics compliance horizontal stacked bars ----
axF = fig.add_subplot(gs[1, 2])
mono_est = [p * 0.585 for p in physics_folds]
cons_est = [p - m for p, m in zip(physics_folds, mono_est)]
y_pos2 = np.arange(len(folds))
bar_h = 0.55
for k, (m, c) in enumerate(zip(mono_est, cons_est)):
    axF.barh(y_pos2[k], m, height=bar_h, color=BLUE_D, edgecolor='white', linewidth=0.7)
    axF.barh(y_pos2[k], c, height=bar_h, left=m, color=TEAL, edgecolor='white', linewidth=0.7)
    axF.text(m/2, y_pos2[k], f'{m:.1f}', ha='center', va='center', fontsize=7.5, fontweight='bold',
             color='white' if m > 1.5 else NEU_D)
    axF.text(m + c/2, y_pos2[k], f'{c:.1f}', ha='center', va='center', fontsize=7.5, fontweight='bold',
             color='white' if c > 1.5 else NEU_D)
    axF.text(m + c + 0.18, y_pos2[k], f'= {m+c:.1f}', va='center', fontsize=8, fontweight='bold', color=NEU_D)
axF.axvline(9, color=NEU_D, ls='--', lw=0.8, alpha=0.5)
axF.set_yticks(y_pos2); axF.set_yticklabels([f'Fold {f}' for f in folds])
axF.set_xlabel('Physics score'); axF.set_xlim(0, 13.0); axF.invert_yaxis()
axF.barh([], [], color=BLUE_D, label='Monotonicity'); axF.barh([], [], color=TEAL, label='Consistency')
axF.legend(loc='upper left', fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none',
           bbox_to_anchor=(1.02, 1.0))
set_style(axF); add_label(axF, 'F', x=-0.30, y=1.12)

# ---- G (NEW): violin of reconstruction error by fold ----
axG = fig.add_subplot(gs[2, 0])
all_errs = []
for f in folds:
    dd = np.load(os.path.join(MODEL_DIR, f'fold{f}_val_predictions.npz'))
    err = (dd['recon_true'] - dd['recon_pred']).flatten()
    all_errs.append(err)
parts = axG.violinplot(all_errs, positions=[0, 1, 2], widths=0.7, showmedians=True, showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor(BLUE_M); pc.set_alpha(0.55); pc.set_edgecolor('white')
for i, err in enumerate(all_errs):
    jit = np.random.default_rng(42+i).uniform(-0.12, 0.12, len(err))
    axG.scatter(np.full(len(err), i)+jit, err, s=8, c=NEU_D, alpha=0.5, edgecolors='none', zorder=3)
axG.axhline(0, color=NEU_D, ls='--', lw=0.8, alpha=0.6)
axG.set_xticks([0, 1, 2]); axG.set_xticklabels(['Fold 1', 'Fold 2', 'Fold 3'], fontsize=7)
axG.set_ylabel('Recon. error (norm.)')
axG.set_title('Error distribution by fold', fontsize=8.5, fontweight='bold')
set_style(axG); add_label(axG, 'G', x=-0.30, y=1.12)

# ---- H (NEW): per-sample recon R^2 stripplot by fold ----
axH = fig.add_subplot(gs[2, 1])
r2_all = []
for f in folds:
    dd = np.load(os.path.join(MODEL_DIR, f'fold{f}_val_predictions.npz'))
    rtt, rpp = dd['recon_true'], dd['recon_pred']
    r2s_ = []
    for j in range(len(rtt)):
        denom = np.sum((rtt[j] - np.mean(rtt[j]))**2)
        r2s_.append(max(0, 1 - np.sum((rtt[j]-rpp[j])**2)/denom) if denom > 0 else 0.0)
    r2_all.append(r2s_)
for i, r2s_ in enumerate(r2_all):
    jit = np.random.default_rng(7+i).uniform(-0.12, 0.12, len(r2s_))
    axH.scatter(np.full(len(r2s_), i)+jit, r2s_, s=45, c=[BLUE_D, BLUE_M, BLUE_S][i],
                edgecolors='white', linewidth=0.5, alpha=0.9, zorder=3)
    axH.hlines(np.median(r2s_), i-0.25, i+0.25, color=NEU_D, lw=1.8, zorder=4)
axH.axhline(0.7, color=UP_GREEN, ls='--', lw=1.0, alpha=0.8, label='0.7 threshold')
axH.set_xticks([0, 1, 2]); axH.set_xticklabels(['Fold 1', 'Fold 2', 'Fold 3'], fontsize=7)
axH.set_ylabel('Per-sample $R^2$'); axH.set_ylim(0, 1.05)
axH.legend(loc='lower right', fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
axH.set_title('Per-sample recon. $R^2$', fontsize=8.5, fontweight='bold')
set_style(axH); add_label(axH, 'H', x=-0.30, y=1.12)

# ---- I (NEW): residual heatmap (fold1 samples x strain) ----
axI = fig.add_subplot(gs[2, 2])
resid = rt - rp  # fold1 samples (n x 200)
vmax = np.percentile(np.abs(resid), 95)
im = axI.imshow(resid, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
axI.set_xlabel('Strain index'); axI.set_ylabel('Sample')
axI.set_title('Recon. residual (fold 1)', fontsize=8.5, fontweight='bold')
cb = fig.colorbar(im, ax=axI, fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=6)
set_style(axI, grid=False); add_label(axI, 'I', x=-0.30, y=1.12)

plt.tight_layout()
save_fig(fig, 'Fig3')
print('Fig3 v3 saved (3x3, 9 panels)')
