# TransMICRO-Net

**TransMICRO-Net — Transformer for Multi-material Inter-system Cross-mechanism
Representation and Outcome**

A quasi-multimodal, multi-task deep learning framework that simultaneously
reconstructs full stress-strain curves and predicts peak stress, yield point,
and toughness of biomaterial hydrogels from limited compression data.

This repository accompanies the manuscript
*"TransMICRO-Net: A Quasi-multimodal Multi-task Interpretable Attention-based
Deep Learning Framework for Accurate Prediction and Cross-mechanism
Deciphering of Mechanical Fingerprints in Biomaterial Hydrogels"* (Frontiers).
The full case-study analysis and additional figures are provided in the
Supplementary Material of that article.

---

## Highlights

- **Quasi-multimodal inputs** — 200-point stress sequence + material identity +
  concentration, conditioned via FiLM layers.
- **Convolutional residual backbone + multi-head self-attention** for long-range
  strain-axis dependencies.
- **Uncertainty-weighted multi-task loss** (Kendall et al. 2018) balancing curve
  reconstruction and property regression.
- **Snapshot ensembling + Mixup + Gaussian noise augmentation** for the
  data-scarce regime (18 training specimens, 3-fold CV).
- **Interpretability toolbox** — attention fingerprinting, gradient sensitivity
  mapping, latent-space (UMAP/PCA) analysis, permutation importance, and
  partial dependence.

## Repository layout

```
TransMICRO-Net/
├── train.py                 # main training script (3-fold CV + snapshot ensemble + external test)
├── preprocess.py            # extract compression curves from raw Zenodo data into standard tensors
├── downstream_analysis.py   # gradient sensitivity, reconstruction diagnostics, embeddings, cycle stability
├── plotting.py              # all manuscript figures (scatter, violin, attention maps, UMAP, etc.)
├── processed_data/          # preprocessed tensors (curves, labels, metadata)
├── requirements.txt
└── README.md
```

## Data

The raw experimental dataset is publicly available at Zenodo
(record 18171138; DOI 10.5281/zenodo.18171138), originally provided by
Faber et al. It contains uniaxial compression cyclic tests of OHA-GEL,
Alginate, and ADA-GEL hydrogels. `preprocess.py` converts the raw archive
into the tensors stored in `processed_data/`.

## Installation

```bash
git clone https://github.com/SHENTongfei/TransMICRO-Net.git
cd TransMICRO-Net
pip install -r requirements.txt
```

Python 3.9+ recommended; a CUDA GPU is optional (CPU works, slower).

## Usage

```bash
# 1. (optional) rebuild processed tensors from the raw Zenodo archive
python preprocess.py

# 2. train TransMICRO-Net (3-fold CV + snapshot ensemble + external ADA-GEL test)
python train.py

# 3. downstream interpretability analyses
python downstream_analysis.py

# 4. regenerate manuscript figures
python plotting.py
```

Outputs are written under `models/`, `logs/`, and `analysis/` (created
automatically).

## Key results (manuscript)

- Mean cross-validated R²: **0.824 ± 0.049** (3-fold, 3 indicators).
- External test on unseen ADA-GEL: **0.896** overall; yield point 0.991 and
  toughness 0.974, top-ranked among all compared models.
- Physics compliance score: 14.95 / 15 on the external test set.

## License

For research use. Contact the corresponding authors for details.

## Citation

If you use this code or data, please cite:

- Shen, T. et al. *TransMICRO-Net: A Quasi-multimodal Multi-task Interpretable
  Attention-based Deep Learning Framework...* (2026).
- Faber, J. et al. *Experimental data of OHA-GEL, ADA-GEL and alginate
  hydrogels for hyperelastic parameter identification.* Zenodo,
  DOI: 10.5281/zenodo.18171138.
