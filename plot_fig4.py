"""SCI-style Fig 4: external test (A-C scatter, D recon, E err hist, F mean err, G physics)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sci_common.py'), encoding='utf-8').read())

from sklearn.preprocessing import StandardScaler
y_train_raw = np.load(os.path.join(DATA_DIR, 'train_val_labels.npy'))
scaler_y = StandardScaler()
scaler_y.fit(np.log1p(y_train_raw))
X_train_raw = np.load(os.path.join(DATA_DIR, 'train_val_curves.npy'))
scaler_X = StandardScaler()
scaler_X.fit(X_train_raw)

recon_scaled = np.load(os.path.join(DOWN_DIR, 'ext_recon_scaled.npy'))
preds_log_std = np.load(os.path.join(DOWN_DIR, 'ext_preds_log.npy'))
preds_real = np.expm1(scaler_y.inverse_transform(preds_log_std))
recon_real = scaler_X.inverse_transform(recon_scaled)

fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.4))
axes = axes.flatten()

models4 = ['TransMICRO-Net', 'RF', 'DT', 'KNN']
colors4 = [C_OURS, BLUE_D, BLUE_M, NEU_M]
markers4 = ['o', 's', '^', 'D']
preds_dict = {'TransMICRO-Net': pred_ext, 'RF': base_preds['RF'], 'DT': base_preds['DT'], 'KNN': base_preds['KNN']}

# A-C: scatter with joint limits
for i in range(3):
    ax = axes[i]
    for mname, c, mk in zip(models4, colors4, markers4):
        p = preds_dict[mname]
        ax.scatter(true_ext[:, i], p[:, i], s=42, c=c, marker=mk, edgecolors='white',
                   linewidth=0.6, label=mname, zorder=5, alpha=0.92)
    allv = np.concatenate([true_ext[:, i]] + [preds_dict[m][:, i] for m in models4])
    lo, hi = allv.min(), allv.max()
    pad = (hi-lo)*0.12 if hi > lo else 1.0
    lim = [lo-pad, hi+pad]
    ax.plot(lim, lim, 'k--', lw=0.8, alpha=0.45)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f'Measured {targets[i]} ({units[i]})')
    ax.set_ylabel(f'Predicted {targets[i]} ({units[i]})')
    ax.legend(loc='lower right', fontsize=6, frameon=True, fancybox=True, framealpha=0.85, edgecolor='none')
    set_style(ax); add_label(ax, chr(65+i))
    ax.text(0.03, 0.97, 'n = 3', transform=ax.transAxes, fontsize=7, va='top',
            color=NEU_D, style='italic')

# D: external recon example
ax = axes[3]
true_c = smooth_curve(X_test[0])
pred_c = smooth_curve(recon_real[0])
ax.plot(strain, true_c, '-', color='#272727', lw=1.6, label='Measured')
ax.plot(strain, pred_c, '--', color=C_OURS, lw=1.5, label='Reconstructed')
ax.fill_between(strain, true_c, pred_c, alpha=0.12, color=C_OURS)
ax.set_xlabel('Strain'); ax.set_ylabel('Stress (Pa)')
ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.85, edgecolor='none', fontsize=6)
set_style(ax); add_label(ax, 'D')

# E: recon error histogram + normal fit
ax = axes[4]
errs = (X_test - recon_real).flatten()
ax.hist(errs, bins=22, color=BLUE_D, alpha=0.75, edgecolor='white', linewidth=0.5, density=True)
mu, sd = np.mean(errs), np.std(errs)
if sd > 0:
    xs = np.linspace(mu-3*sd, mu+3*sd, 200)
    ax.plot(xs, 1/(sd*np.sqrt(2*np.pi))*np.exp(-(xs-mu)**2/(2*sd**2)), '-', color=C_OURS, lw=1.4, label='Normal fit')
ax.axvline(0, color=NEU_D, ls='--', lw=0.8)
ax.set_xlabel('Reconstruction error')
ax.set_ylabel('Density')
ax.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.85, edgecolor='none')
set_style(ax); add_label(ax, 'E')

# F: mean recon error profile
ax = axes[5]
mean_err = np.mean(recon_real - X_test, axis=0)
std_err = np.std(recon_real - X_test, axis=0)
ax.plot(strain, mean_err, '-', color=C_OURS, lw=1.5, label='Mean error')
ax.fill_between(strain, mean_err-std_err, mean_err+std_err, alpha=0.18, color=C_OURS, label='±1 SD')
ax.axhline(0, color=NEU_D, ls='--', lw=0.8)
ax.set_xlabel('Strain'); ax.set_ylabel('Error (Pa)')
ax.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.85, edgecolor='none')
set_style(ax); add_label(ax, 'F')

# G: physics compliance
ax = axes[6]
ax.bar(['Monotonicity', 'Consistency', 'Total'], [8.0, 6.95, 14.95],
       color=[BLUE_D, TEAL, C_OURS], width=0.6, edgecolor='white', linewidth=0.8)
for xi, v in enumerate([8.0, 6.95, 14.95]):
    ax.text(xi, v+0.3, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
ax.set_ylabel('Score (max 15)')
ax.set_ylim(0, 16.8)
set_style(ax); add_label(ax, 'G')

axes[7].set_visible(False)
plt.tight_layout()
save_fig(fig, 'Fig4')
