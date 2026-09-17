# Quantum Corrosion Classification

**Hybrid Quantum-Classical Pipeline for Radar-IQ Oxide Quantification**

Code accompanying *"Quantum Machine Learning for Radar-Based Corrosion Monitoring"* (Denis, Sammartino, Di Pietro). Classifies steel oxidation state (quantity, deposition geometry, and oxide composition) from harmonic radar IQ measurements using a hybrid quantum-classical architecture, benchmarked against classical baselines (Random Forest, XGBoost, LightGBM, ResNet, MobileNetV3, EfficientNet).

## Architecture

```
Raw IQ (I/Q samples)
  → 512-bin Power Spectral Density (Welch/FFT)
  → Z-score standardization + statistical/phase features (1040-d)
  → PCA (256-d, ~63% retained variance)
      ├── Quantum path:  Amplitude encoding (8 qubits) → 6-layer strongly-entangling VQC (144 params)
      │                  → Pauli-Z expectation readout (8-d)
      └── Classical path: Dense(256→128) → ReLU → Dropout
  → Concatenate (136-d) → classification head (oxide class) + regression head (mass, MAE/Huber)
```

## Repository contents

This repository contains the **code only**. Raw IQ recordings, processed datasets, and trained model checkpoints are not included here — see [Data Availability](#data-availability) below.

```
.
├── qnn_v6.py, qnn_corrosion.py     # Hybrid VQC model definitions and training entry points
├── train_hybrid_models.py          # Trains the EfficientNet/ResNet/MobileNet + VQC variants
├── run_ablation_experiments.py     # VQC-only / CNN-only / encoding & depth ablations
├── run_ablation_gaussian.py        # Noise-robustness ablation
├── eval_ensemble.py, quick_ensemble_eval.py, test_ensemble_setup.py
│                                    # Ensemble evaluation utilities
├── check_inputs.py                 # Sanity checks on IQ / feature inputs
├── run_quick_corrosion.py, run_iris_smoke.py
│                                    # Smoke-test / quick-run entry points
├── generate_eval_notebooks.py      # Programmatically generates the per-experiment notebooks below
├── src/                            # Core library
│   ├── config.py                   # Hyperparameters
│   ├── preprocessing.py            # IQ → PSD → PCA feature pipeline
│   ├── dataset.py                  # PyTorch datasets, stratified splits
│   ├── classical_models.py         # ResNet / MobileNet / EfficientNet backbones
│   ├── quantum_models.py           # PennyLane hybrid VQC (amplitude embedding, ring entanglement)
│   ├── training.py                 # Training engine (AMP, gradient clipping, early stopping)
│   ├── evaluation.py                # Metrics computation & export
│   └── plots.py                    # Figure generation (t-SNE, saliency, confusion matrices, ...)
├── scripts/                        # Pipeline orchestration and standalone experiment runners
│   ├── run_pipeline.py             # Main orchestrator (preprocess/classical/quantum/ablation/figures)
│   ├── run_hybrid_sota.py, run_hybrid_cnn_qnn_99.py, run_vqc_fewshot_99.py
│   └── quantum_pure_over90_search.py
├── notebooks/                      # Cross-dataset generalization checks (PMLB, FinTSB)
├── *.ipynb (root)                  # Per-experiment analysis notebooks (corrosion merging, stack/spread
│                                    # geometry, saliency and t-SNE analysis, ablations, baselines)
└── paper/                          # LaTeX source of the accompanying manuscript
    ├── article.tex
    └── bibliography.bib
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the harmonic radar IQ dataset (see Data Availability) and place it under data/raw/
data/raw/
├── 0.5/    ← .iq binary files (0.5 g oxide mass)
├── 1/
├── 1.5/
├── 2/
└── 2.5/

# 3. Run the full pipeline
python -m scripts.run_pipeline --data-dir data/raw --device cuda

# Or run individual stages:
python -m scripts.run_pipeline --stage preprocess
python -m scripts.run_pipeline --stage classical --epochs 150 --rounds 10
python -m scripts.run_pipeline --stage quantum
python -m scripts.run_pipeline --stage ablation
python -m scripts.run_pipeline --stage figures
```

### CLI options (`run_pipeline.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--stage` | `all` | Pipeline stage: preprocess/classical/quantum/efficiency/ablation/figures |
| `--data-dir` | `data/raw` | Path to raw IQ data |
| `--device` | auto | `cuda` or `cpu` |
| `--epochs` | 150 | Training epochs |
| `--rounds` | 10 | Independent training rounds |
| `--batch-size` | 32 | Batch size |

## Key design decisions

- **Amplitude encoding**: 256-d PCA features are mapped exactly onto the amplitudes of an 8-qubit register (2^8 = 256), avoiding zero-padding.
- **6-layer strongly-entangling ansatz**: 3×8×6 = 144 trainable quantum parameters, with a cyclic CNOT ring entangling topology.
- **Parameter-shift gradients**: quantum-circuit gradients are computed via the parameter-shift rule for hardware compatibility; state-vector simulation is used for the reported results (ideal, noiseless).
- **Multi-task objective**: joint classification (cross-entropy, label smoothing) and regression (Huber loss) heads share the fused 136-d latent representation.
- **Gradient saliency**: input-space gradients are projected back through the PCA loading matrix onto the original 512-bin PSD to produce physically interpretable saliency maps.
- **Capacity-matched baselines**: at low data fractions the hybrid model is compared against Random Forest / XGBoost / LightGBM rather than only large CNNs, to isolate the effect of the quantum branch from raw parameter-count mismatch.

## Data Availability

The harmonic radar IQ recordings supporting the findings are available at [10.5281/zenodo.18171674](https://doi.org/10.5281/zenodo.18171674).

## Citation

If you use this code, please cite the accompanying manuscript (see `paper/article.tex`):

```
Denis, N., Sammartino, V., Di Pietro, R.
"Quantum Machine Learning for Radar-Based Corrosion Monitoring."
IEEE Transactions on Quantum Engineering.
```

## License

No license file is currently included; contact the authors before reuse.
