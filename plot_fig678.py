"""SCI-style Fig 6-8 (deduplicated, journal-quality with NMI palette)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sci_common.py'), encoding='utf-8').read())

DOWN = os.path.join(ANALYSIS_DIR, 'downstream')

# ================= Fig 6: attention + permutation =================
attn_train = np.load(os.path.join(DOWN, 'attn_weights_train.npy'))
attn_labels = np.load(os.path.join(DOWN, 'attn_labels_train.npy'))
perm_imp = np.load(os.path.join(DOWN, 'permutation_importance.npy'))
perm_seg = np.load(os.path.join(DOWN, 'permutation_segments.npy'))

fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
# A: mean attention deviation with shaded band
ax = axes[0]
for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
    mask = attn_labels == mat
    if mask.sum() == 0: continue
    dev = (attn_train[mask] - 0.005) * 1e3
    ax.plot(strain, dev.mean(axis=0), color=color, lw=1.6, label=name)
    ax.fill_between(strain, dev.mean(axis=0)-dev.std(axis=0), dev.mean(axis=0)+dev.std(axis=0),
                    color=color, alpha=0.15)
ax.axhline(0, color=NEU_D, ls='--', lw=0.7, alpha=0.6)
ax.set_xlabel('Strain')
ax.set_ylabel('Attention deviation (×$10^{-3}$)')
ax.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.85, edgecolor='none')
set_style(ax); add_label(ax, 'A')

# B: per-sample traces (faint)
ax = axes[1]
for i in range(len(attn_train)):
    color = C_OHA if attn_labels[i] == 0 else C_ALG
    ax.plot(strain, (attn_train[i]-0.005)*1e3, color=color, lw=0.5, alpha=0.35)
ax.axhline(0, color=NEU_D, ls='--', lw=0.7, alpha=0.6)
ax.set_xlabel('Strain'); ax.set_ylabel('Attention deviation (×$10^{-3}$)')
set_style(ax); add_label(ax, 'B')

# C: permutation importance (log-scale, grouped bars)
ax = axes[2]
segs = perm_seg[:8]
imps = perm_imp[:8, :]
x = np.arange(len(segs))
w = 0.26
for j, (tname, tcolor) in enumerate(zip(targets, [C_OURS, C_OHA, C_ALG])):
    ax.bar(x + (j-1)*w, imps[:, j], width=w, color=tcolor, alpha=0.88,
           edgecolor='white', linewidth=0.5, label=tname)
ax.set_yscale('symlog', linthresh=10)
ax.set_xticks(x)
ax.set_xticklabels([f'S{i}' for i in range(len(segs))], fontsize=6, rotation=45)
ax.set_ylabel('ΔMSE (symlog)')
ax.legend(fontsize=6, ncol=3, frameon=True, fancybox=True, framealpha=0.85, edgecolor='none')
set_style(ax); add_label(ax, 'C')

plt.tight_layout()
save_fig(fig, 'Fig6')

# ================= Fig 7: latent + PDP + cycle =================
from sklearn.manifold import TSNE
latent_train = np.load(os.path.join(DOWN, 'latent_train.npy'))
latent_test = np.load(os.path.join(DOWN, 'latent_test.npy'))
latent_labels_tr = np.load(os.path.join(DOWN, 'latent_labels_train.npy'))
latent_labels_te = np.load(os.path.join(DOWN, 'latent_labels_test.npy'))
pdp_oha = np.load(os.path.join(DOWN, 'pdp_oha.npz'))

try:
    from umap import UMAP
    reducer = UMAP(n_neighbors=8, min_dist=0.2, random_state=42)
    z_tr = reducer.fit_transform(latent_train)
    z_te = reducer.transform(latent_test)
except Exception:
    reducer = TSNE(n_components=2, random_state=42, perplexity=5)
    z_all = reducer.fit_transform(np.vstack([latent_train, latent_test]))
    z_tr, z_te = z_all[:len(latent_train)], z_all[len(latent_train):]

fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
# A: UMAP with soft edges
ax = axes[0]
for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
    mask = latent_labels_tr == mat
    ax.scatter(z_tr[mask, 0], z_tr[mask, 1], s=50, c=color, alpha=0.85,
               edgecolors='white', linewidth=0.6, label=name, zorder=3)
ax.scatter(z_te[:, 0], z_te[:, 1], s=80, marker='D', c=C_ADA, alpha=0.95,
           edgecolors='white', linewidth=0.8, label='ADA-GEL (test)', zorder=5)
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
ax.legend(fontsize=6, loc='best', frameon=True, fancybox=True, framealpha=0.85, edgecolor='none')
set_style(ax); add_label(ax, 'A')

# B: PDP peak stress
ax = axes[1]
conc = pdp_oha['conc']
preds = pdp_oha['preds']
pd_vals = preds[:, 0] if preds.ndim == 2 else preds
ax.plot(conc, pd_vals, '-o', color=C_OURS, lw=1.6, ms=4, mfc='white', mew=1.2)
ax.set_xlabel('Concentration (%)')
ax.set_ylabel('Predicted peak stress (Pa)')
set_style(ax); add_label(ax, 'B')

# C: cycle stability
ax = axes[2]
csv = os.path.join(DOWN, 'cycle_stability.csv')
if os.path.exists(csv):
    cd = pd.read_csv(csv)
    for (mat, conc1), g in cd.groupby(['material', 'conc1']):
        pred = dict(zip(g['cycle'], g['pred_peak_stress']))
        if 1 in pred and 3 in pred:
            color = C_OHA if mat == 'OHA-GEL' else C_ALG
            marker = 'o' if mat == 'OHA-GEL' else 's'
            ax.scatter(pred[1], pred[3], s=55, c=color, marker=marker,
                       edgecolors='white', linewidth=0.6, alpha=0.9,
                       label=f'{mat} {conc1}%' if conc1 in (2.5, 3.75) else None)
    ax.plot([100, 1200], [100, 1200], 'k--', lw=0.8, alpha=0.45)
    ax.set_xlim(100, 1200); ax.set_ylim(100, 1200)
    ax.set_xlabel('Cycle 1 peak stress (Pa)')
    ax.set_ylabel('Cycle 3 peak stress (Pa)')
    ax.legend(fontsize=5, loc='best', frameon=True, fancybox=True, framealpha=0.85, edgecolor='none')
set_style(ax); add_label(ax, 'C')

plt.tight_layout()
save_fig(fig, 'Fig7')

# ================= Fig 8: gradient + recon error =================
gs = np.load(os.path.join(DOWN, 'gradient_sensitivity.npz'))
sens = gs['sensitivity']
sens_labels = gs['labels'] if 'labels' in gs else None
rd = np.load(os.path.join(DOWN, 'recon_error_diagnosis.npz'))
errs = rd['errors']
err_mats = rd['materials']

fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
# A: mean sensitivity by material with band
ax = axes[0]
if sens.ndim == 2 and sens_labels is not None:
    for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
        mask = sens_labels == mat
        if mask.sum() == 0: continue
        ax.plot(strain, sens[mask].mean(axis=0), color=color, lw=1.6, label=name)
        ax.fill_between(strain, sens[mask].mean(axis=0)-sens[mask].std(axis=0),
                        sens[mask].mean(axis=0)+sens[mask].std(axis=0), color=color, alpha=0.15)
ax.set_xlabel('Strain'); ax.set_ylabel('Sensitivity')
ax.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.85, edgecolor='none')
set_style(ax); add_label(ax, 'A')

# B: sensitivity heatmap (robust vmax)
ax = axes[1]
if sens.ndim == 2:
    vmax = np.percentile(np.abs(sens), 95)
    im = ax.imshow(sens, aspect='auto', cmap='Reds', vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_xlabel('Strain index'); ax.set_ylabel('Sample')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6)
set_style(ax, grid=False); add_label(ax, 'B')

# C: per-sample mean |err| by material (stripplot-style)
ax = axes[2]
err_mag = np.abs(errs).mean(axis=1)
for mat, color, name in [(0, C_OHA, 'OHA-GEL'), (1, C_ALG, 'Alginate')]:
    mask = err_mats == mat
    vals = err_mag[mask]
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
    ax.scatter(np.full(len(vals), 0)+jitter, vals, s=34, c=color, alpha=0.85,
               edgecolors='white', linewidth=0.6, label=name, zorder=3)
    ax.hlines(np.median(vals), -0.25, 0.25, color=color, lw=1.6, zorder=4)
ax.set_xticks([0, 1]); ax.set_xticklabels(['OHA-GEL', 'Alginate'], fontsize=7)
ax.set_ylabel('Mean |recon error|')
ax.legend(fontsize=6, loc='upper right', frameon=True, fancybox=True, framealpha=0.85, edgecolor='none')
set_style(ax); add_label(ax, 'C')

plt.tight_layout()
save_fig(fig, 'Fig8')
print('ALL SCI FIGS DONE')
