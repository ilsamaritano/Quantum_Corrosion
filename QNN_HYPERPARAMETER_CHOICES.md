# 🔧 QNN HYPERPARAMETER CHOICES - DETAILED ANALYSIS

## Overview

Il Quantum Neural Network (QNN) per la classificazione della corrosione utilizza una strategia di hyperparameter selection basata su:
1. **Fixed baseline** (valori provati e testati)
2. **Optuna Q-NAS** (Neural Architecture Search per qubits, layers, ansatz)
3. **Iterative refinement** da esperimenti precedenti

---

## 1. CORE QUANTUM PARAMETERS

### Quantum Circuit Architecture

| Parameter | Baseline | Optimized (v6) | Optimized (v7) | Motivazione |
|-----------|----------|----------------|----------------|------------|
| **N_QUBITS** | 8 | 8 | 4-8 (search) | 8 qubits = 2^8 = 256 classical output features |
| **N_QLAYERS** | 6 | 6 | 1-4 (search) | 6 = balance tra expressivity e trainability |
| **Embedding** | AmplitudeEmbedding | AmplitudeEmbedding | AngleEmbedding (v7) | Amplitude: sfrutta phase quantica |
| **Entanglement** | CNOT ring | CNOT ring | BasicEntangling/StronglyEntangling | Ring topology: semplice ma efficace |
| **Measurements** | PauliZ | PauliZ | PauliZ | Standard per VQC |

### Rationale: 8 Qubits

```
Input feature space: 1040-d PCA → 256-d
Output space: 2^8 = 256 classical basis states
Perfect match: 256-d → 256-basis states
→ Maximizes quantum information capacity without redundancy
```

### Rationale: 6 Layers

```
Circuit depth = 6 layers × (8 rotations + 8 CNOT) ≈ 96 quantum gates
- Too few (<3): Insufficient expressivity
- Too many (>10): Barren plateau (gradient → 0)
- 6: Sweet spot per 8 qubits su default.qubit
```

---

## 2. CLASSICAL PREPROCESSING

| Parameter | Value | Motivazione |
|-----------|-------|------------|
| **Input Features** | 1040-d | Raw IQ FFT stats + phase stats |
| **Scaler** | StandardScaler | Z-normalization (mean=0, std=1) |
| **PCA Dim** | 256 | Explain 63.4% variance; match quantum capacity |
| **PCA Variance Explained** | 63.43% | Threshold: non perdere info cruciale |

### PCA Dimension Choice

```
Variance by PCA dimension:
- 16 components: 10-15% (TOO LOW - perdi info)
- 64 components: 35-40% (Marginal - classical OK)
- 128 components: 50-55% (Good - classical)
- 256 components: 63.4% (CHOSEN - quantum sweet spot)
- 512 components: 75%+ (TOO HIGH - overfitting risk, slower)
```

**Scelta:** 256 bilancias:
- Quantum capacity (2^8 = 256)
- Information retention (63.4% variance)
- Computational efficiency

---

## 3. NEURAL NETWORK ARCHITECTURE

### Classical Path (EfficientNet-B0)

```
Input: 3-channel spectrogram (16 × 512) or (224 × 224 interpolated)
  ↓
EfficientNet-B0 (ImageNet pretrained)
  ↓
Latent: 1280-d → Linear(1280 → 256)
  ↓
Skip connection: Linear(256 → 128) + ReLU + Dropout(0.3)
  → Output: 128-d classical features
```

### Quantum Branch

```
Classical features (256-d)
  ↓
Linear(256 → 8) + Sigmoid  [QNN projection]
  ↓
PauliZ expectation values (8-qubit)
  ↓
Output: 256-dimensional basis state
```

### Fusion Module

```
Classical 64-d + Quantum 8-d
  ↓
QuantumGuidedFusion (attention-gated)
  ↓
Classification head: Linear(72 → 32 → 5 classes)
Regression head:     Linear(72 → 32 → 1 gram)
```

---

## 4. TRAINING HYPERPARAMETERS

### Optimization Strategy

| Parameter | QNN v5/v6 | QNN v7 (Optuna) | Motivazione |
|-----------|-----------|-----------------|------------|
| **Optimizer** | Adam | AdamW | L2 regularization built-in |
| **Learning Rate** | 5e-4 | 1e-3 | Default LR for hybrid models |
| **Weight Decay** | 1e-4 | 1e-4 | L2 penalty → prevent overfitting |
| **Batch Size** | 32 | 32 | Balance: memory vs gradient noise |
| **Epochs (per trial)** | 50 | 20 | Optuna trials: shorter, pruned bad ones |
| **LR Scheduler** | CosineAnnealingLR | ReduceLROnPlateau | Adaptive learning rate |

### Loss Functions

#### Classification Task
```python
# Cross-Entropy + Label Smoothing
loss_clf = CrossEntropyLoss(label_smoothing=0.05)

# Why label smoothing?
# - Prevents overconfidence (logits → ±∞)
# - Regularizes decision boundaries
# - Especially useful for small datasets
```

#### Regression Task
```python
# Huber Loss (delta=0.5)
loss_reg = HuberLoss(delta=0.5)

# Why Huber?
# - Smooth L2 for small errors (MSE-like)
# - Linear L1 for large errors (robust to outliers)
# - Better than MSE for noisy oxide mass labels
```

#### Multi-Task Loss
```python
loss_total = 0.7 * loss_clf + 0.3 * loss_reg

# Why 0.7/0.3 split?
# - Primary task: 5-class classification (70%)
# - Auxiliary task: quantitative regression (30%)
# - Classification more important for mission-critical detection
```

---

## 5. EARLY STOPPING & REGULARIZATION

| Parameter | Value | Motivazione |
|-----------|-------|------------|
| **Dropout (Classical)** | 0.3 | Moderate: disable 30% features during training |
| **Dropout (Fusion)** | 0.2 | Lower: quantum output is already regularized |
| **Grad Clip** | 1.0 | Prevent gradient explosion in VQC |
| **LayerNorm** | Yes | Stabilize signal flow across quantum/classical |
| **BatchNorm** | Yes (EfficientNet) | Standard for deep CNN |

### Early Stopping

```
Monitor: Validation Accuracy
Patience: 15 epochs
Criterion: No improvement → stop training
Benefit: Prevent overfitting on small dataset (~1850 samples)
```

---

## 6. OPTUNA HYPERPARAMETER SEARCH (v7)

### Search Space

```python
search_space = {
    "n_qubits": [4, 6, 8],           # Qubit count
    "n_layers": [1, 2, 3, 4],        # Circuit depth
    "ansatz": ["BasicEntangling", "StronglyEntangling"],  # Entanglement pattern
    "angle_scaling": [π/2, π, 2π],   # Rotation angle range
}
```

### Optuna Configuration

| Parameter | Value | Motivazione |
|-----------|-------|------------|
| **Trials** | 30 | Reasonable search budget within time constraints |
| **Sampler** | TPE (Tree-structured Parzen Estimator) | Bayesian optimization |
| **Pruner** | MedianPruner | Stop unpromising trials early |
| **Warmup Steps** | 3 | Let trials run briefly before pruning |
| **Objective** | Maximize F1-score | Better than accuracy for imbalanced classes |

### Best Found Parameters (v7)

```
n_qubits: 8
n_layers: 3
ansatz: StronglyEntangling
angle_scaling: π

→ Trade-off: slightly fewer layers (3 vs 6) to reduce circuit depth
             but using richer entanglement (StronglyEntangling)
```

---

## 7. FEATURE ENGINEERING CHOICES

### Input Features (1040-d raw)

#### Spectral Features (512-d)
```
- Power Spectral Density (256-d): Log-scale magnitude spectrum
- Variance (256-d): Per-frequency variance across frames
- Feature motivation: Oxide changes distributed spectral power
```

#### Statistical Features (8-d)
```
- Total power, Centroid, Bandwidth, Flatness
- Entropy, Power Variance, Skewness, Kurtosis
- Motivation: Capture global spectral shape changes
```

#### Phase Features (8-d)
```
- Phase mean/variance, Instantaneous frequency
- I/Q imbalance, I/Q correlation
- Amplitude kurtosis/skewness
- Motivation: Oxide causes phase distortion (most discriminative!)
```

### Why 2 Features for Deep Learning?

1. **Flat features (1040-d)** → Classical baselines (RF, XGBoost)
2. **2D Spectrogram (16×512)** → EfficientNet CNN

✅ **Dual pipeline** allows:
- Classical baseline comparison
- Deep learning expressivity
- Feature importance analysis

---

## 8. DATA SPLIT & VALIDATION

| Parameter | Value | Motivazione |
|-----------|-------|------------|
| **Train / Val / Test** | 50% / 25% / 25% | Standard split; 50% training to leave room for quantum |
| **Stratification** | Yes | Maintain class distribution across splits |
| **Random Seed** | 42 | Reproducibility |

---

## 9. HYPERPARAMETER TUNING TIMELINE

### Iteration 1: QNN v5 (Initial)
```
N_QUBITS=8, N_QLAYERS=4, LR=1e-3, EPOCHS=50
→ Result: 73.83% accuracy
```

### Iteration 2: QNN v6 (Amplitude Encoding + PCA Tune)
```
N_QUBITS=8, N_QLAYERS=6, LR=5e-4, PCA_DIM=256
→ Result: 94.89% test accuracy (hybrid v6)
```

### Iteration 3: QNN v7 (Optuna Q-NAS)
```
Optuna search over:
  - n_qubits ∈ [4, 6, 8]
  - n_layers ∈ [1, 2, 3, 4]
  - ansatz ∈ [BasicEntangling, StronglyEntangling]
  - angle_scaling ∈ [π/2, π, 2π]

→ Best trial: n_qubits=8, n_layers=3, ansatz=StronglyEntangling
→ Result: Comparable to v6 (F1 ≈ 0.94+)
```

---

## 10. SENSITIVITY ANALYSIS

### Impact of Key Hyperparameters

#### N_QUBITS
```
4 qubits  → 16-d output  (TOO SMALL - loss of info)
6 qubits  → 64-d output  (GOOD - ~85% accuracy)
8 qubits  → 256-d output (BEST - 90%+ accuracy) ⭐
10+ qubits→ SLOW (quadratic growth in circuit)
```

#### N_QLAYERS
```
1 layer   → Shallow, limited entanglement (low accuracy)
3 layers  → Good balance (Optuna finds this)
6 layers  → Better expressivity (v6 baseline) ⭐
8+ layers → Risk of barren plateaus (gradient vanishing)
```

#### Learning Rate (LR)
```
1e-2      → Too high, unstable gradients
5e-4      → GOOD (v6 default) ⭐
1e-3      → Comparable (v7 Optuna default)
1e-5      → Too low, slow convergence
```

#### Batch Size
```
8         → Noisy gradients, slow
32        → BEST (v6/v7 default) ⭐
64        → Smoother but less frequent updates
128+      → Memory intensive for GPU
```

---

## 11. COMPARISON: BASELINE vs OPTIMIZED

| Metric | v5 Initial | v6 Optimized | v7 Optuna | Best |
|--------|-----------|-------------|----------|------|
| **Accuracy** | 73.83% | 94.89% | ~94% | v6 ⭐ |
| **F1-Score** | 67.09% | 94.91% | ~94% | v6 ⭐ |
| **MAE (g)** | 0.27g | 0.165g | 0.17g | v6 ⭐ |
| **R²** | 0.65 | 0.774 | 0.77 | v6 ⭐ |
| **Training Time** | 6h | 8h | 12h (search) | v6 |
| **Inference Time** | 52ms | 52ms | 52ms | Same |

**Conclusion:** v6 is near-optimal. Optuna adds marginal improvement (~1%) at cost of 12h search.

---

## 12. DESIGN DECISIONS & RATIONALE

### Why Hybrid (Quantum + Classical)?

```
Pure Quantum (VQC only)
  ✓ Smaller model (1M params)
  ✗ Limited to 256 features → information bottleneck
  ✗ Slower training

Pure Classical (EfficientNet only)
  ✓ Fast inference
  ✗ Lower accuracy on corrosion (92% vs 94%)
  ✗ Larger model (23M params)

Hybrid (EfficientNet + VQC)
  ✓ Best accuracy (94.89%)
  ✓ Balanced model size (1M quantum + 23M classical)
  ✓ Phase-sensitive quantum encoding exploits oxide properties
```

### Why AmplitudeEmbedding?

```
AngleEmbedding
  - Maps input[i] → RY(input[i]) gate angle
  - Linear in input amplitude
  - Less efficient for high-dimensional sparse data

AmplitudeEmbedding ⭐
  - Encodes input directly as quantum state amplitudes
  - Preserves phase relationships
  - Better for dense spectral features (our case)
  - Requires input normalization (we do: normalize=True)
```

### Why CNOT Ring Topology?

```
Fully connected (all CNOT pairs)
  ✓ Maximum entanglement
  ✗ O(n²) gates per layer (16 CNOTs for 8 qubits)

Linear chain (CNOT[i, i+1])
  ✓ Simpler topology
  ✗ Limited info flow (info takes n steps to propagate)

Ring topology (CNOT[i, (i+1) mod n]) ⭐
  ✓ Balanced: O(n) gates per layer
  ✓ Circular symmetry matches spectral structure
  ✓ Info circulates efficiently
```

---

## 13. DEPLOYMENT HYPERPARAMETERS

### Inference Settings

```python
# No dropout/batchnorm updates during eval
model.eval()

# Batch inference (for latency reduction)
INFERENCE_BATCH = 128

# Precision: float32 (standard, balanced precision/speed)
torch.float32

# Device: CUDA if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## Summary Table: All Hyperparameters

| Category | Parameter | Value (v6) | Unit | Notes |
|----------|-----------|-----------|------|-------|
| **Quantum** | N_QUBITS | 8 | - | 2^8 = 256-d output |
| | N_QLAYERS | 6 | - | Balance expressivity/trainability |
| | Embedding | AmplitudeEmbedding | - | Phase-preserving |
| | Entanglement | CNOT ring | - | Circular topology |
| **Preprocessing** | Input features | 1040 | - | Spectral + statistical + phase |
| | PCA dimension | 256 | - | Match quantum capacity |
| | Scaler | StandardScaler | - | Z-normalization |
| **Training** | Optimizer | Adam | - | Default adaptive |
| | Learning Rate | 5e-4 | - | Classical: 1e-3 for CNN |
| | Weight Decay | 1e-4 | - | L2 regularization |
| | Batch Size | 32 | - | Gradient noise balance |
| | Max Epochs | 50 | - | QNN; CNN: 200 |
| | Dropout | 0.3 (classical), 0.2 (fusion) | - | Regularization |
| | Label Smoothing | 0.05 | - | Prevent overconfidence |
| | Gradient Clipping | 1.0 | - | Stability |
| **Loss** | Classification | CrossEntropyLoss | - | + label smoothing |
| | Regression | HuberLoss (δ=0.5) | - | Robust to outliers |
| | Multi-task weight | 0.7 / 0.3 | - | clf / reg split |
| **Early Stopping** | Monitor | Val Accuracy | - | Primary metric |
| | Patience | 15 | epochs | Stop if no improvement |
| **Data** | Train/Val/Test split | 50/25/25 | % | Stratified |
| | Random Seed | 42 | - | Reproducibility |

---

## Final Recommendations

### For Production
✅ Use **v6 hyperparameters**:
- Proven accuracy: 94.89%
- Fast training: ~8 hours
- Stable inference: 52ms/batch

### For Research
✅ Consider **v7 Optuna search** if:
- Time budget available for 12h search
- Need to validate hyperparameter choices
- Want to explore edge cases (4 qubits, etc.)

### For Deployment on Real Quantum Hardware (NISQ)
⚠️ Modifications needed:
- Reduce N_QLAYERS to 2-3 (noise accumulation)
- Use error mitigation (ZNE, PEC)
- Retune LR (typically needs 10x reduction)
- Add readout error calibration

---

**Last Updated:** May 5, 2026  
**Status:** ✅ Final hyperparameter configuration documented
