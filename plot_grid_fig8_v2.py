"""SCI Fig 8 v2: Hero panels + sensitivity/attention dual-axis (advanced form)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sci_common.py'), encoding='utf-8').read())

from matplotlib.gridspec import GridSpec

DOWN = os.path.join(ANALYSIS_DIR, 'downstream')
sens_npz = np.load(os.path.join(DOWN, 'gradient_sensitivity.npz'))
sens = sens_npz['sensitivity']
sens_labels = sens_npz['labels'] if 'labels' in sens_npz.files else None
rd = np.load(os.path.join(DOWN, 'recon_error_diagnosis.npz'))
errs = rd['errors']
err_mats = rd['materials']
attn_train = np.load(os.path.join(DOWN, 'attn_weights_train.npy'))
attn_labels = np.load(os.path.join(DOWN, 'attn_labels_train.npy'))

fig = plt.figure(figsize=(10.4, 7.6))
grid = GridSpec(3, 3, hspace=0.55, wspace=0.45)

# A: HERO - sensitivity + attention dual-axis (OHA-GEL)
axA = fig.add_subplot(grid[0, 0])
if sens.ndim == 2 and sens_labels is not None:
    mask_o = sens_labels == 0
    axA.plot(strain, sens[mask_o].mean(axis=0), color=C_OURS, lw=2.0, label='Sensitivity')
    mask_a = attn_labels == 0
    if mask_a.sum() > 0:
        attn_dev = (attn_train[mask_a].mean(axis=0) - 0.005) * 1e3
        axA2 = axA.twinx()
        axA2.plot(strain, attn_dev, color=BLUE_D, lw=1.5, ls='--', alpha=0.8, label='Attention (x1e-3)')
        axA2.set_ylabel('Attention deviation (x$10^{-3}$)', fontsize=8, color=BLUE_D)
        axA2.tick_params(axis='y', colors=BLUE_D, labelsize=7)
        axA2.set_ylim(-3, 6)
axA.axvline(0, color=NEU_D, ls='--', lw=0.8, alpha=0.6)
axA.set_xlabel('Strain'); axA.set_ylabel('Sensitivity (OHA-GEL)', color=C_OURS)
axA.tick_params(axis='y', colors=C_OURS, labelsize=7)
axA.legend(loc='upper right', fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axA); add_label(axA, 'A', x=-0.14, y=1.06)
axA.set_title('Hero: sensitivity + attention (OHA)', fontsize=9, fontweight='bold')

# B: HERO companion (Alginate)
axB = fig.add_subplot(grid[0, 1])
mask_a_sens = sens_labels == 1
axB.plot(strain, sens[mask_a_sens].mean(axis=0), color=C_OURS, lw=2.0, label='Sensitivity')
mask_a_attn = attn_labels == 1
if mask_a_attn.sum() > 0:
    attn_dev = (attn_train[mask_a_attn].mean(axis=0) - 0.005) * 1e3
    axB2 = axB.twinx()
    axB2.plot(strain, attn_dev, color=BLUE_D, lw=1.5, ls='--', alpha=0.8, label='Attention (x1e-3)')
    axB2.set_ylabel('Attention deviation (x$10^{-3}$)', fontsize=8, color=BLUE_D)
    axB2.tick_params(axis='y', colors=BLUE_D, labelsize=7)
    axB2.set_ylim(-3, 6)
axB.axvline(0, color=NEU_D, ls='--', lw=0.8, alpha=0.6)
axB.set_xlabel('Strain'); axB.set_ylabel('Sensitivity (Alginate)', color=C_OURS)
axB.tick_params(axis='y', colors=C_OURS, labelsize=7)
axB.legend(loc='upper right', fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axB); add_label(axB, 'B', x=-0.14, y=1.06)
axB.set_title('Hero: sensitivity + attention (Alg)', fontsize=9, fontweight='bold')

# C: full sensitivity heatmap
axC = fig.add_subplot(grid[0, 2])
vmax = np.percentile(np.abs(sens), 95)
im = axC.imshow(sens, aspect='auto', cmap='Reds', vmin=0, vmax=vmax, interpolation='nearest')
axC.set_xlabel('Strain index'); axC.set_ylabel('Sample')
cb = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.02)
cb.ax.tick_params(labelsize=6)
set_style(axC, grid=False); add_label(axC, 'C', x=-0.14, y=1.06)

# D: per-sample |err| stripplot
axD = fig.add_subplot(grid[1, 0])
err_mag = np.abs(errs).mean(axis=1)
for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
    mask = err_mats == mat
    vals = err_mag[mask]
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
    axD.scatter(np.full(len(vals), 0)+jitter, vals, s=34, c=color, alpha=0.85,
                edgecolors='white', linewidth=0.6, label=name, zorder=3)
    axD.hlines(np.median(vals), -0.25, 0.25, color=color, lw=1.6, zorder=4)
axD.set_xticks([0, 1]); axD.set_xticklabels(['OHA-GEL', 'Alginate'], fontsize=7)
axD.set_ylabel('Mean |recon error|')
axD.legend(fontsize=6, loc='upper right', frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axD); add_label(axD, 'D', x=-0.14, y=1.06)

# E: residual density by material
axE = fig.add_subplot(grid[1, 1])
for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
    mask = err_mats == mat
    vals = errs[mask].flatten()
    axE.hist(vals, bins=20, alpha=0.5, color=color, edgecolor='white', density=True, label=name)
axE.axvline(0, color=NEU_D, ls='--', lw=0.8)
axE.set_xlabel('Residual'); axE.set_ylabel('Density')
axE.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axE); add_label(axE, 'E', x=-0.14, y=1.06)

# F: combined residual histogram with Normal fit
axF = fig.add_subplot(grid[1, 2])
errs_flat = errs.flatten()
axF.hist(errs_flat, bins=30, color=BLUE_D, alpha=0.75, edgecolor='white', linewidth=0.5)
mu, sd = np.mean(errs_flat), np.std(errs_flat)
if sd > 0:
    xs = np.linspace(mu-3*sd, mu+3*sd, 200)
    axF.plot(xs, 1/(sd*np.sqrt(2*np.pi))*np.exp(-(xs-mu)**2/(2*sd**2)), '-', color=C_OURS, lw=1.4, label='Normal fit')
axF.axvline(0, color=NEU_D, ls='--', lw=0.8)
axF.set_xlabel('Residual'); axF.set_ylabel('Density')
axF.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axF); add_label(axF, 'F', x=-0.14, y=1.06)

# G: per-strain mean residual
axG = fig.add_subplot(grid[2, 0])
for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
    mask = err_mats == mat
    m = errs[mask].mean(axis=0)
    axG.plot(strain, m, color=color, lw=1.5, label=name)
axG.axhline(0, color=NEU_D, ls='--', lw=0.8)
axG.set_xlabel('Strain'); axG.set_ylabel('Mean residual')
axG.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axG); add_label(axG, 'G', x=-0.14, y=1.06)

# H: MAE/RMSE by material (grouped)
axH = fig.add_subplot(grid[2, 1])
metrics = ['MAE', 'RMSE']
o_v = [np.abs(errs[err_mats==0]).mean(), np.sqrt((errs[err_mats==0]**2).mean())]
a_v = [np.abs(errs[err_mats==1]).mean(), np.sqrt((errs[err_mats==1]**2).mean())]
x = np.arange(2); w = 0.35
axH.bar(x-w/2, o_v, w, color=C_OHA, alpha=0.9, edgecolor='white', label='OHA-GEL')
axH.bar(x+w/2, a_v, w, color=C_ALG, alpha=0.9, edgecolor='white', label='Alginate')
for xi, v in enumerate(o_v):
    axH.text(xi-w/2, v+0.005, f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')
for xi, v in enumerate(a_v):
    axH.text(xi+w/2, v+0.005, f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')
axH.set_xticks(x); axH.set_xticklabels(metrics)
axH.set_ylabel('Error')
axH.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axH); add_label(axH, 'H', x=-0.14, y=1.06)

# I: VIOLIN + stripplot (combined, "advanced" form)
axI = fig.add_subplot(grid[2, 2])
parts_o = axI.violinplot([err_mag[err_mats==0]], positions=[0.0], widths=0.5, showmeans=False, showmedians=True)
parts_a = axI.violinplot([err_mag[err_mats==1]], positions=[1.0], widths=0.5, showmeans=False, showmedians=True)
for pc in parts_o['bodies']:
    pc.set_facecolor(C_OHA); pc.set_alpha(0.5); pc.set_edgecolor('white')
for pc in parts_a['bodies']:
    pc.set_facecolor(C_ALG); pc.set_alpha(0.5); pc.set_edgecolor('white')
axI.scatter(np.full((err_mats==0).sum(), 0.0), err_mag[err_mats==0], s=18,
            c=C_OHA, edgecolors='white', linewidth=0.4, alpha=0.7)
axI.scatter(np.full((err_mats==1).sum(), 1.0), err_mag[err_mats==1], s=18,
            c=C_ALG, edgecolors='white', linewidth=0.4, alpha=0.7)
axI.set_xticks([0, 1]); axI.set_xticklabels(['OHA-GEL', 'Alginate'], fontsize=7)
axI.set_ylabel('Mean |recon error| per sample')
set_style(axI); add_label(axI, 'I', x=-0.14, y=1.06)

plt.tight_layout()
save_fig(fig, 'Fig8')
print('Fig8 v2 saved')