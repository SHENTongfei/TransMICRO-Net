"""SCI-style Fig 8 (3x3 grid): gradient sensitivity + recon error + diagnostics"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sci_common.py'), encoding='utf-8').read())

from matplotlib.gridspec import GridSpec

DOWN = os.path.join(ANALYSIS_DIR, 'downstream')
gs = np.load(os.path.join(DOWN, 'gradient_sensitivity.npz'))
sens = gs['sensitivity']
sens_labels = gs['labels'] if 'labels' in gs else None
rd = np.load(os.path.join(DOWN, 'recon_error_diagnosis.npz'))
errs = rd['errors']
err_mats = rd['materials']

fig = plt.figure(figsize=(10.4, 7.6))
grid = GridSpec(3, 3, hspace=0.5, wspace=0.42)

# A: mean sensitivity (hero)
axA = fig.add_subplot(grid[0, 0])
if sens.ndim == 2 and sens_labels is not None:
    for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
        mask = sens_labels == mat
        if mask.sum() == 0: continue
        axA.plot(strain, sens[mask].mean(axis=0), color=color, lw=1.8, label=name)
        axA.fill_between(strain, sens[mask].mean(axis=0)-sens[mask].std(axis=0),
                        sens[mask].mean(axis=0)+sens[mask].std(axis=0), color=color, alpha=0.18)
axA.set_xlabel('Strain'); axA.set_ylabel('Sensitivity')
axA.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axA)
add_label(axA, 'A', x=-0.14, y=1.06)
axA.set_title('Hero: gradient sensitivity', fontsize=9, fontweight='bold')

# B: sensitivity heatmap
axB = fig.add_subplot(grid[0, 1])
vmax = np.percentile(np.abs(sens), 95)
im = axB.imshow(sens, aspect='auto', cmap='Reds', vmin=0, vmax=vmax, interpolation='nearest')
axB.set_xlabel('Strain index'); axB.set_ylabel('Sample')
cb = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.02)
cb.ax.tick_params(labelsize=6)
set_style(axB, grid=False); add_label(axB, 'B', x=-0.14, y=1.06)

# C: per-sample mean |err| stripplot
axC = fig.add_subplot(grid[0, 2])
err_mag = np.abs(errs).mean(axis=1)
for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
    mask = err_mats == mat
    vals = err_mag[mask]
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
    axC.scatter(np.full(len(vals), 0)+jitter, vals, s=34, c=color, alpha=0.85,
                edgecolors='white', linewidth=0.6, label=name, zorder=3)
    axC.hlines(np.median(vals), -0.25, 0.25, color=color, lw=1.6, zorder=4)
axC.set_xticks([0, 1]); axC.set_xticklabels(['OHA-GEL', 'Alginate'], fontsize=7)
axC.set_ylabel('Mean |recon error|')
axC.legend(fontsize=6, loc='upper right', frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axC); add_label(axC, 'C', x=-0.14, y=1.06)

# D: sensitivity vs strain per material (overlay)
axD = fig.add_subplot(grid[1, 0])
# Show ridge plot style
offset = 0.005
for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
    mask = sens_labels == mat if sens_labels is not None else np.zeros(len(sens), dtype=bool)
    if mask.sum() == 0: continue
    axD.plot(strain, sens[mask].mean(axis=0) - offset if mat == 1 else sens[mask].mean(axis=0),
            color=color, lw=1.5, label=name)
axD.set_xlabel('Strain'); axD.set_ylabel('Sensitivity (offset)')
axD.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axD); add_label(axD, 'D', x=-0.14, y=1.06)

# E: gradient by material (heatmap)
axE = fig.add_subplot(grid[1, 1])
# show only first 6 samples for visibility
sens_sub = sens[:6] if sens.shape[0] >= 6 else sens
vmax = np.percentile(np.abs(sens_sub), 95)
im = axE.imshow(sens_sub, aspect='auto', cmap='Reds', vmin=0, vmax=vmax, interpolation='nearest')
axE.set_xlabel('Strain index'); axE.set_ylabel('Sample (first 6)')
cb = fig.colorbar(im, ax=axE, fraction=0.046, pad=0.02)
cb.ax.tick_params(labelsize=6)
set_style(axE, grid=False); add_label(axE, 'E', x=-0.14, y=1.06)

# F: error distribution per material (density overlay)
axF = fig.add_subplot(grid[1, 2])
for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
    mask = err_mats == mat
    vals = errs[mask].flatten()
    axF.hist(vals, bins=20, alpha=0.5, color=color, edgecolor='white', density=True, label=name)
axF.axvline(0, color=NEU_D, ls='--', lw=0.8)
axF.set_xlabel('Residual')
axF.set_ylabel('Density')
axF.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axF); add_label(axF, 'F', x=-0.14, y=1.06)

# G: error histogram (combined)
axG = fig.add_subplot(grid[2, 0])
errs_flat = (errs).flatten()
axG.hist(errs_flat, bins=30, color=BLUE_D, alpha=0.75, edgecolor='white', linewidth=0.5)
mu, sd = np.mean(errs_flat), np.std(errs_flat)
if sd > 0:
    xs = np.linspace(mu-3*sd, mu+3*sd, 200)
    axG.plot(xs, 1/(sd*np.sqrt(2*np.pi))*np.exp(-(xs-mu)**2/(2*sd**2)), '-', color=C_OURS, lw=1.4, label='Normal fit')
axG.axvline(0, color=NEU_D, ls='--', lw=0.8)
axG.set_xlabel('Residual')
axG.set_ylabel('Density')
axG.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axG); add_label(axG, 'G', x=-0.14, y=1.06)

# H: per-strain mean error by material
axH = fig.add_subplot(grid[2, 1])
for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
    mask = err_mats == mat
    m = errs[mask].mean(axis=0)
    axH.plot(strain, m, color=color, lw=1.5, label=name)
axH.axhline(0, color=NEU_D, ls='--', lw=0.8)
axH.set_xlabel('Strain'); axH.set_ylabel('Mean residual')
axH.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axH); add_label(axH, 'H', x=-0.14, y=1.06)

# I: error metrics summary
axI = fig.add_subplot(grid[2, 2])
metrics = ['MAE', 'RMSE']
o_v = [np.abs(errs[err_mats==0]).mean(), np.sqrt((errs[err_mats==0]**2).mean())]
a_v = [np.abs(errs[err_mats==1]).mean(), np.sqrt((errs[err_mats==1]**2).mean())]
x = np.arange(2); w = 0.35
axI.bar(x-w/2, o_v, w, color=C_OHA, alpha=0.9, edgecolor='white', label='OHA-GEL')
axI.bar(x+w/2, a_v, w, color=C_ALG, alpha=0.9, edgecolor='white', label='Alginate')
axI.set_xticks(x); axI.set_xticklabels(metrics)
axI.set_ylabel('Error')
axI.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.9, edgecolor='none')
set_style(axI); add_label(axI, 'I', x=-0.14, y=1.06)

plt.tight_layout()
save_fig(fig, 'Fig8')