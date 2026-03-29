# QuantumCorrosion

**Hybrid Quantum-Classical ML Pipeline for Radar IQ Corrosion Classification**

This project implements a complete machine learning pipeline that combines classical
deep learning (ResNet-34, MobileNetV2, SimpleCNN) with a hybrid quantum-classical
classifier (PennyLane VQC) to classify corrosion severity from radar IQ signals.

---

## Project Overview

Radar IQ signals are processed into STFT spectrograms and fed into classical CNN
baselines. For the quantum path, hand-crafted tabular features are extracted and
compressed with PCA before being passed into a variational quantum circuit (VQC)
wrapped in a `nn.Module` for hybrid training with PyTorch.

### Corrosion Classes

| Label | Class |
|-------|-------|
| 0 | `healthy` |
| 1 | `light_corrosion` |
| 2 | `moderate_corrosion` |
| 3 | `severe_corrosion` |

---

## Repository Structure

```
QuantumCorrosion/
├── configs/
│   ├── dataset.yaml          # Dataset and preprocessing config
│   ├── model_classical.yaml  # Classical model hyperparameters
│   ├── model_quantum.yaml    # Quantum model hyperparameters
│   └── experiments.yaml      # Learning curves, ablation, robustness
├── data/
│   ├── raw/                  # Raw .npy IQ files (ignored by git)
│   ├── processed/            # Preprocessed spectrograms & features
│   └── splits/               # Train/val/test splits
├── notebooks/
│   ├── eda.ipynb             # Exploratory data analysis
│   └── sanity_checks.ipynb   # Preprocessing pipeline sanity checks
├── results/
│   ├── confusion_matrices/   # Confusion matrix .npy arrays
│   ├── figures/              # Generated paper figures (PNG)
│   ├── metrics/              # JSON/CSV metric files
│   └── models/               # Saved model checkpoints
├── scripts/
│   ├── prepare_data.py       # Data loading, validation, preprocessing
│   ├── train_classical.py    # Train ResNet-34/MobileNetV2/SimpleCNN
│   ├── train_quantum.py      # Train hybrid quantum classifier
│   ├── run_ablation.py       # Ablation study (n_qubits x n_layers x encoding)
│   └── generate_figures.py   # Generate all 12 paper figures
├── src/
│   ├── classical_models.py   # ResNet-34, MobileNetV2, SimpleCNN, MLP
│   ├── data_loading.py       # IQ loading, validation, splitting
│   ├── dimensionality_reduction.py  # PCA + MLP autoencoder
│   ├── evaluation.py         # Inference, timing, memory
│   ├── metrics.py            # Classification metrics, CSV I/O
│   ├── plots.py              # All 12 paper figures
│   ├── preprocessing.py      # FFT, spectrograms, augmentation
│   ├── quantum_models.py     # HybridQuantumClassifier (PennyLane + PyTorch)
│   ├── spectrograms.py       # STFT, mel, resize, tensor conversion
│   ├── training.py           # Training loop, early stopping, learning curves
│   └── utils.py              # Seeding, logging, JSON I/O
├── tests/
│   ├── test_data_loading.py
│   ├── test_metrics.py
│   ├── test_models.py
│   └── test_preprocessing.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

### Prerequisites

- Python >= 3.9
- pip

### Setup

```bash
git clone https://github.com/your-org/QuantumCorrosion.git
cd QuantumCorrosion
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** PennyLane and `pennylane-lightning` may require additional system
> dependencies. See [PennyLane docs](https://docs.pennylane.ai) for details.

---

## Quick Start

### 1. Prepare Data

```bash
# Generate synthetic data and prepare splits (no real data needed)
python scripts/prepare_data.py --synthetic --seed 42
```

To use real data, place `<class_name>.npy` files (shape `(N, signal_length)`)
in `data/raw/` and run without `--synthetic`:

```bash
python scripts/prepare_data.py --data_dir data/raw --output_dir data/splits
```

### 2. Train Classical Models

```bash
# Train all classical models
python scripts/train_classical.py --model all --device cpu

# Train a specific model
python scripts/train_classical.py --model resnet34 --device cuda
```

### 3. Train Quantum Model

```bash
python scripts/train_quantum.py \
    --data_dir data/splits/tabular \
    --n_qubits 4 --n_layers 3 --encoding angle
```

### 4. Run Ablation Study

```bash
python scripts/run_ablation.py --n_epochs 20
```

### 5. Generate Paper Figures

```bash
python scripts/generate_figures.py --results_dir results --output_dir results/figures
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Experiments

### Learning Curves

Controlled by `configs/experiments.yaml`. Trains each model on increasing
fractions of the training set and reports mean +/- std over multiple seeds.

### Ablation Study

Sweeps over:
- `n_qubits` in {2, 4, 6, 8}
- `n_layers` in {1, 2, 3, 4, 5}
- `encoding` in {angle, amplitude}

Results saved to `results/metrics/ablation_results.csv`.

### Robustness Analysis

Evaluates model accuracy under increasing levels of input Gaussian noise
(sigma in {0.0, 0.01, 0.05, 0.1, 0.2, 0.5}).

---

## Architecture: Hybrid Quantum Classifier

```
Input Features (16-dim PCA)
        |
  Linear(16 -> n_qubits) + Tanh
        |
  Angle Encoding (RY rotations)
        |
  +-----------------------------+
  | Variational Quantum Layer   | x n_layers
  | RY(theta) + RZ(phi) / qubit |
  | CNOT ring entanglement      |
  +-----------------------------+
        |
  Measurement: probs(2^n_qubits)
        |
  Linear(2^n_qubits -> n_classes)
        |
  Class Logits
```

---

## Paper Figures

| Figure | Description |
|--------|-------------|
| fig1_pipeline.png | End-to-end pipeline diagram |
| fig2_signals_spectrograms.png | Raw IQ signals + STFT spectrograms |
| fig3_hybrid_architecture.png | Hybrid architecture diagram |
| fig4_quantum_circuit.png | VQC circuit schematic |
| fig5_metrics_comparison.png | Accuracy/F1/precision/recall comparison |
| fig6_param_comparison.png | Trainable parameter counts (log scale) |
| fig7_learning_curves.png | Validation accuracy vs training fraction |
| fig8_confusion_matrices.png | Normalised confusion matrices |
| fig9_accuracy_vs_complexity.png | Accuracy vs parameter count scatter |
| fig10_ablation_study.png | Ablation heatmap + bar chart |
| fig11_robustness.png | Accuracy vs noise level |
| fig12_runtime.png | Inference time comparison |

---

## Configuration Reference

### `configs/dataset.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `n_classes` | 4 | Number of corrosion classes |
| `signal_length` | 512 | IQ signal length |
| `n_samples_per_class` | 200 | Samples per class (synthetic) |
| `synthetic` | true | Use synthetic data |

### `configs/model_quantum.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `n_qubits` | 4 | Number of qubits |
| `n_layers` | 3 | VQC depth |
| `encoding` | angle | Encoding: `angle` or `amplitude` |
| `latent_dim` | 16 | PCA components |

---

## License

MIT License.
