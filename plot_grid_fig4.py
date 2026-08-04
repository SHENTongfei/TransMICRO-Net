"""SCI-style Fig 4 (3x3 grid): external test on ADA-GEL
Hero: A (peak stress scatter, large). B-C: yield/toughness scatter.
D: recon example. E: error hist. F: mean err profile. G: physics. H: MAE by model. I: R2 by model.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sci_common.py'), encoding='utf-8').read())

from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler

y_train_raw = np.load(os.path.join(DATA_DIR, 'train_val_labels.npy'))
scaler_y = StandardScaler(); scaler_y.fit(np.log1p(y_train_raw))
X_train_raw = np.load(os.path.join(DATA_DIR, 'train_val_curves.npy'))
scaler_X = StandardScaler(); scaler_X.fit(X_train_raw)
recon_scaled = np.load(os.path.join(DOWN_DIR, 'ext_recon_scaled.npy'))
preds_log_std = np.load(os.path.join(DOWN_DIR, 'ext_preds_log.npy'))
preds_real = np.expm1(scaler_y.inverse_transform(preds_log_std))
recon_real = scaler_X.inverse_transform(recon_scaled)

fig = plt.figure(figsize=(10.8, 7.6))
gs = GridSpec(3, 3, hspace=0.5, wspace=0.42)

models4 = ['TransMICRO-Net', 'RF', 'DT', 'KNN']
colors4 = [C_OURS, BLUE_D, BLUE_M, NEU_M]
markers4 = ['o', 's', '^', 'D']
preds_dict = {'TransMICRO-Net': pred_ext, 'RF': base_preds['RF'], 'DT': base_preds['DT'], 'KNN': base_preds['KNN']}

# A-C: scatter panels (A is hero)
for i in range(3):
    ax = fig.add_subplot(gs[0, i])
    for mname, c, mk in zip(models4, colors4, markers4):
        p = preds_dict[mname]
        ax.scatter(true_ext[:, i], p[:, i], s=48, c=c, marker=mk, edgecolors='white',
                   linewidth=0.6, label=mname, zorder=5, alpha=0.92)
    allv = np.concatenate([true_ext[:, i]] + [preds_dict[m][:, i] for m in models4])
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
        ax.set_title('Peak stress — hero', fontsize=9, fontweight='bold')

# D: external recon example
ax = fig.add_subplot(gs[1, 0])
true_c = smooth_curve(X_test[0])
pred_c = smooth_curve(recon_real[0])
ax.plot(strain, true_c, '-', color='#272727', lw=1.6, label='Measured')
ax.plot(strain, pred_c, '--', color=C_OURS, lw=1.5, label='Reconstructed')
ax.fill_between(strain, true_c, pred_c, alpha=0.12, color=C_OURS)
ax.set_xlabel('Strain'); ax.set_ylabel('Stress (Pa)')
ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9, edgecolor='none', fontsize=6)
set_style(ax); add_label(ax, 'D', x=-0.14, y=1.06)

# E: recon error histogram
ax = fig.add_subplot(gs[1, 1])
errs = (X_test - recon_real).flatten()
ax.hist(errs, bins=22, color=BLUE_D, alpha=0.75, edgecolor='white', linewidth=0.5, density=True)
mu, sd = np.mean(errs), np.std(errs)
if sd > 0:
    xs = np.linspace(mu-3*sd, mu+3*sd, 200)
    ax.plot(xs, 1/(sd*np.sqrt(2*np.pi))*np.exp(-(xs-mu)**2/(2*sd**2)), '-', color=C_OURS, lw=1.4, label='Normal fit')
ax.axvline(0, color=NEU_D, ls='--', lw=0.8)
ax.set_xlabel('Reconstruction error'); ax.set_ylabel('Density')
ax.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(ax); add_label(ax, 'E', x=-0.14, y=1.06)

# F: mean recon error profile
ax = fig.add_subplot(gs[1, 2])
mean_err = np.mean(recon_real - X_test, axis=0)
std_err = np.std(recon_real - X_test, axis=0)
ax.plot(strain, mean_err, '-', color=C_OURS, lw=1.5, label='Mean error')
ax.fill_between(strain, mean_err-std_err, mean_err+std_err, alpha=0.18, color=C_OURS, label='±1 SD')
ax.axhline(0, color=NEU_D, ls='--', lw=0.8)
ax.set_xlabel('Strain'); ax.set_ylabel('Error (Pa)')
ax.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(ax); add_label(ax, 'F', x=-0.14, y=1.06)

# G: physics compliance
ax = fig.add_subplot(gs[2, 0])
ax.bar(['Mono.', 'Consist.', 'Total'], [8.0, 6.95, 14.95],
       color=[BLUE_D, TEAL, C_OURS], width=0.6, edgecolor='white', linewidth=0.8)
for xi, v in enumerate([8.0, 6.95, 14.95]):
    ax.text(xi, v+0.3, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
ax.set_ylabel('Score (max 15)')
ax.set_ylim(0, 16.8)
set_style(ax); add_label(ax, 'G', x=-0.14, y=1.06)

# H: MAE by model (overall)
ax = fig.add_subplot(gs[2, 1])
mae_ov = [mae['Ours']['overall'], mae['RF']['overall'], mae['DT']['overall'], mae['KNN']['overall']]
bars = ax.bar(models4, mae_ov, color=colors4, alpha=0.9, width=0.6, edgecolor='white', linewidth=0.8)
for b, v in zip(bars, mae_ov):
    ax.text(b.get_x()+b.get_width()/2, v+0.6, f'{v:.2f}', ha='center', fontsize=7, fontweight='bold')
ax.set_ylabel('Overall MAE')
ax.set_ylim(0, max(mae_ov)*1.3)
ax.tick_params(axis='x', labelsize=6.5)
set_style(ax); add_label(ax, 'H', x=-0.14, y=1.06)

# I: per-indicator MAE comparison (grouped)
ax = fig.add_subplot(gs[2, 2])
x = np.arange(3)
w = 0.2
models_mae = ['Ours', 'RF', 'DT', 'KNN']
for j, (mn, mc) in enumerate(zip(models_mae, colors4)):
    vals = [mae[mn]['peak'], mae[mn]['yield']*20, mae[mn]['tough']]  # yield scaled for visibility
    ax.bar(x + (j-1.5)*w, vals, width=w, color=mc, alpha=0.85, edgecolor='white', linewidth=0.5,
           label=mn)
ax.set_xticks(x); ax.set_xticklabels(['Peak', 'Yield×20', 'Tough'])
ax.set_ylabel('MAE (Pa / scaled)')
ax.legend(fontsize=5.5, ncol=2, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(ax); add_label(ax, 'I', x=-0.14, y=1.06)

plt.tight_layout()
save_fig(fig, 'Fig4')
