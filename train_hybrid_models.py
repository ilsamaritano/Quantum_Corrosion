#!/usr/bin/env python3
"""
Hybrid Quantum-Classical Model Comparison
==========================================
Compares VQC combined with:
  - ResNet
  - MobileNet
  - EfficientNet
  - Baseline MLP (from qnn_v6)

All models use the same feature extraction pipeline and training setup.
"""

import os, sys, time, json, re, warnings
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.signal import get_window
from scipy.stats import skew, kurtosis as scipy_kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              mean_absolute_error, r2_score, confusion_matrix)
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pennylane as qml
from pennylane import numpy as pnp

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/sammarv/quantum_corrosion")
IQ_FILES = {
    0: BASE_DIR / "data/raw/0.5/0.5.iq",
    1: BASE_DIR / "data/raw/1/1.iq",
    2: BASE_DIR / "data/raw/1.5/1.5.iq",
    3: BASE_DIR / "data/raw/2/2.iq",
    4: BASE_DIR / "data/raw/2.5/2.5.iq",
}
GRAM_VALUES = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5}
SPLITS_DIR = BASE_DIR / "data/splits"
OUT_DIR = BASE_DIR / "results/hybrid_models_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Signal config ─────────────────────────────────────────────────────────────
FFT_SIZE = 4096
N_STACKS = 16
OVERLAP = 0.25
HOP = int(FFT_SIZE * (1 - OVERLAP))
SAMPLES_PER_IMG = N_STACKS * HOP + (FFT_SIZE - HOP)
WIN = get_window("hann", FFT_SIZE).astype(np.float32)

# ── QNN config ────────────────────────────────────────────────────────────────
N_QUBITS   = 8
N_QLAYERS  = 6
PCA_DIM    = 256      # Input dimension for models
N_CLASSES  = 5
TRAIN_FRAC = 0.50
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_WORKERS  = 8

print("=" * 80)
print("Hybrid Quantum-Classical Model Comparison")
print(f"Device: {DEVICE}   Workers: {N_WORKERS}")
print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE EXTRACTION (from qnn_v6.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_img_idx(path: str) -> int:
    m = re.search(r"img(\d+)", str(path))
    return int(m.group(1)) if m else 0


def extract_one(iq: np.memmap, img_idx: int) -> np.ndarray:
    """Extract rich spectral features for one image window from raw IQ."""
    start = img_idx * SAMPLES_PER_IMG
    end   = start + SAMPLES_PER_IMG
    if end > len(iq):
        return None

    frames = np.empty((N_STACKS, FFT_SIZE), dtype="complex64")
    for i in range(N_STACKS):
        s = start + i * HOP
        frames[i] = iq[s : s + FFT_SIZE]

    spec = np.fft.fftshift(np.fft.fft(frames * WIN, axis=1), axes=1)
    mag  = np.abs(spec).astype(np.float32)

    psd_mean = mag.mean(axis=0)
    psd_db   = 20.0 * np.log10(psd_mean + 1e-12)
    psd_512  = psd_db.reshape(512, 8).mean(axis=1)

    psd_var  = mag.var(axis=0)
    var_512  = np.log1p(psd_var.reshape(512, 8).mean(axis=1))

    total_power = float(psd_mean.sum())
    freqs = np.arange(FFT_SIZE, dtype=np.float32)
    centroid = float((freqs * psd_mean).sum() / (psd_mean.sum() + 1e-12))
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * psd_mean).sum() /
                               (psd_mean.sum() + 1e-12)))
    log_psd = np.log(psd_mean + 1e-12)
    flatness = float(np.exp(log_psd.mean()) / (psd_mean.mean() + 1e-12))
    p = psd_mean / (psd_mean.sum() + 1e-12)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    frame_power = mag.sum(axis=1)
    power_var   = float(frame_power.var())
    power_skew  = float(skew(frame_power))
    power_kurt  = float(scipy_kurtosis(frame_power))

    stats = np.array([total_power, centroid, bandwidth, flatness,
                      entropy, power_var, power_skew, power_kurt],
                     dtype=np.float32)

    phase = np.angle(spec).astype(np.float32)
    phase_mean_std  = float(np.std(phase.mean(axis=0)))
    phase_var_mean  = float(phase.var(axis=1).mean())
    inst_freq = np.diff(np.unwrap(phase, axis=0), axis=0)
    if_mean = float(inst_freq.mean())
    if_std  = float(inst_freq.std())
    i_pwr = (frames.real ** 2).mean()
    q_pwr = (frames.imag ** 2).mean()
    iq_imbalance = float(i_pwr / (q_pwr + 1e-12))
    iq_corr = float(np.corrcoef(frames.real.flatten()[:2048],
                                frames.imag.flatten()[:2048])[0, 1])
    amp = np.abs(frames.flatten()[:4096])
    amp_kurt = float(scipy_kurtosis(amp))
    amp_skew = float(skew(amp))

    phase_feats = np.array([phase_mean_std, phase_var_mean, if_mean, if_std,
                             iq_imbalance, iq_corr, amp_kurt, amp_skew],
                            dtype=np.float32)

    return np.concatenate([psd_512, var_512, stats, phase_feats])


def extract_features_parallel(meta_list, iqs, desc="", n_workers=N_WORKERS):
    """Extract features for a list of metadata entries in parallel."""
    n = len(meta_list)
    feats, labels = [None] * n, [None] * n
    t0 = time.time()

    def _worker(i, rec):
        label = rec["label"]
        idx   = _get_img_idx(rec["path"])
        f = extract_one(iqs[label], idx)
        return i, f, label

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker, i, rec): i for i, rec in enumerate(meta_list)}
        done = 0
        for fut in as_completed(futs):
            i, f, lbl = fut.result()
            feats[i]  = f
            labels[i] = lbl
            done += 1

    valid = [(f, l) for f, l in zip(feats, labels) if f is not None]
    X = np.stack([v[0] for v in valid])
    y = np.array([v[1] for v in valid])
    print(f"  Done: {len(valid)} features in {time.time()-t0:.1f}s  shape={X.shape}")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# 2. QUANTUM CIRCUIT
# ═══════════════════════════════════════════════════════════════════════════════

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def vqc(inputs, weights):
    """Variational Quantum Circuit with AmplitudeEmbedding."""
    qml.AmplitudeEmbedding(features=inputs, wires=range(N_QUBITS), normalize=True)
    for layer in range(N_QLAYERS):
        for q in range(N_QUBITS):
            qml.Rot(weights[layer, q, 0],
                    weights[layer, q, 1],
                    weights[layer, q, 2], wires=q)
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLASSICAL MODELS + VQC HYBRIDS
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumHead(nn.Module):
    """Quantum circuit wrapper for hybrid models."""
    def __init__(self):
        super().__init__()
        w_shape = (N_QLAYERS, N_QUBITS, 3)
        self.q_weights = nn.Parameter(
            torch.randn(*w_shape, dtype=torch.float32) * 0.1
        )

    def forward(self, x):
        """x shape: (batch, PCA_DIM) -> (batch, N_QUBITS)"""
        q_results = vqc(x, self.q_weights)
        q_out = torch.stack(q_results, dim=-1).float()
        return q_out


# ────────────────────────────────────────────────────────────────────────────────
# Baseline MLP + VQC
# ────────────────────────────────────────────────────────────────────────────────

class BaselineHybridQNN(nn.Module):
    """Original baseline from qnn_v6.py"""
    def __init__(self):
        super().__init__()
        self.classical_path = nn.Sequential(
            nn.Linear(PCA_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.quantum_head = QuantumHead()
        self.clf_head = nn.Sequential(
            nn.Linear(N_QUBITS + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, N_CLASSES),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(N_QUBITS + 128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        c_out = self.classical_path(x)
        q_out = self.quantum_head(x)
        combined = torch.cat([q_out, c_out], dim=-1)
        logits = self.clf_head(combined)
        grams = self.reg_head(combined).squeeze(1)
        return logits, grams


# ────────────────────────────────────────────────────────────────────────────────
# ResNet-based Hybrid
# ────────────────────────────────────────────────────────────────────────────────

class ResNetBlock(nn.Module):
    """Basic residual block."""
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x):
        residual = x if self.skip is None else self.skip(x)
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = out + residual
        out = F.relu(out)
        return out


class ResNetHybridQNN(nn.Module):
    """ResNet feature extractor + VQC"""
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResNetBlock(PCA_DIM, 256),
            ResNetBlock(256, 256),
            ResNetBlock(256, 128),
            nn.AdaptiveAvgPool1d(1) if False else nn.Identity(),  # no pooling for 1D
        )
        self.feat_project = nn.Linear(128, 128)
        self.quantum_head = QuantumHead()
        self.clf_head = nn.Sequential(
            nn.Linear(N_QUBITS + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, N_CLASSES),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(N_QUBITS + 128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        c_out = self.feature_extractor(x)
        c_out = self.feat_project(c_out)
        q_out = self.quantum_head(x)
        combined = torch.cat([q_out, c_out], dim=-1)
        logits = self.clf_head(combined)
        grams = self.reg_head(combined).squeeze(1)
        return logits, grams


# ────────────────────────────────────────────────────────────────────────────────
# MobileNet-based Hybrid (depthwise separable)
# ────────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparable(nn.Module):
    """Depthwise separable convolution block for 1D features."""
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        # Depthwise
        self.dw = nn.Linear(in_dim, in_dim)
        self.bn1 = nn.BatchNorm1d(in_dim)
        # Pointwise
        self.pw = nn.Linear(in_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.relu(self.bn1(self.dw(x)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.pw(out)))
        return out


class MobileNetHybridQNN(nn.Module):
    """MobileNet-style feature extractor + VQC"""
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            DepthwiseSeparable(PCA_DIM, 256),
            DepthwiseSeparable(256, 256),
            DepthwiseSeparable(256, 128),
        )
        self.feat_project = nn.Linear(128, 128)
        self.quantum_head = QuantumHead()
        self.clf_head = nn.Sequential(
            nn.Linear(N_QUBITS + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, N_CLASSES),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(N_QUBITS + 128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        c_out = self.feature_extractor(x)
        c_out = self.feat_project(c_out)
        q_out = self.quantum_head(x)
        combined = torch.cat([q_out, c_out], dim=-1)
        logits = self.clf_head(combined)
        grams = self.reg_head(combined).squeeze(1)
        return logits, grams


# ────────────────────────────────────────────────────────────────────────────────
# EfficientNet-based Hybrid (inverted residual)
# ────────────────────────────────────────────────────────────────────────────────

class InvertedResidual(nn.Module):
    """Inverted residual block (MBConv style)."""
    def __init__(self, in_dim, out_dim, expand_ratio=6, dropout=0.3):
        super().__init__()
        hidden_dim = int(in_dim * expand_ratio)
        self.expand = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ) if expand_ratio != 1 else nn.Identity()
        self.depthwise = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        self.skip = (in_dim == out_dim)

    def forward(self, x):
        residual = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        if self.skip:
            out = out + residual
        return out


class EfficientNetHybridQNN(nn.Module):
    """EfficientNet-style feature extractor + VQC"""
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            InvertedResidual(PCA_DIM, 256, expand_ratio=6),
            InvertedResidual(256, 256, expand_ratio=6),
            InvertedResidual(256, 128, expand_ratio=6),
        )
        self.feat_project = nn.Linear(128, 128)
        self.quantum_head = QuantumHead()
        self.clf_head = nn.Sequential(
            nn.Linear(N_QUBITS + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, N_CLASSES),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(N_QUBITS + 128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        c_out = self.feature_extractor(x)
        c_out = self.feat_project(c_out)
        q_out = self.quantum_head(x)
        combined = torch.cat([q_out, c_out], dim=-1)
        logits = self.clf_head(combined)
        grams = self.reg_head(combined).squeeze(1)
        return logits, grams


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, opt, clf_loss_fn, reg_loss_fn,
                gram_vals, alpha=0.5, device=DEVICE):
    model.train()
    total_loss = correct = n = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        g_b = torch.tensor([gram_vals[int(l)] for l in y_b.cpu()],
                            dtype=torch.float32, device=device)
        opt.zero_grad()
        logits, grams = model(X_b)
        loss = (1 - alpha) * clf_loss_fn(logits, y_b) + alpha * reg_loss_fn(grams, g_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * len(y_b)
        correct    += (logits.argmax(1) == y_b).sum().item()
        n          += len(y_b)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, gram_vals, device=DEVICE):
    model.eval()
    all_pred, all_true, all_gram_pred, all_gram_true = [], [], [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logits, grams = model(X_b)
        all_pred.extend(logits.argmax(1).cpu().tolist())
        all_true.extend(y_b.cpu().tolist())
        all_gram_pred.extend(grams.cpu().tolist())
        all_gram_true.extend([gram_vals[int(l)] for l in y_b.cpu()])
    acc  = accuracy_score(all_true, all_pred)
    f1   = f1_score(all_true, all_pred, average="macro")
    mae  = mean_absolute_error(all_gram_true, all_gram_pred)
    r2   = r2_score(all_gram_true, all_gram_pred)
    return acc, f1, mae, r2, all_pred, all_true


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model_name, model_class, X_tr_16, y_tr, X_val_16, y_val,
                X_te_16, y_te, gram_vals_map, n_epochs=100, patience=30):
    """Train a single model and return results."""
    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"{'='*80}")

    T = lambda a: torch.from_numpy(a.astype(np.float32))
    tr_ds  = TensorDataset(T(X_tr_16),  torch.from_numpy(y_tr.astype(np.int64)))
    val_ds = TensorDataset(T(X_val_16), torch.from_numpy(y_val.astype(np.int64)))
    te_ds  = TensorDataset(T(X_te_16),  torch.from_numpy(y_te.astype(np.int64)))

    BS = 32
    tr_loader  = DataLoader(tr_ds,  batch_size=BS, shuffle=True,  drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False)
    te_loader  = DataLoader(te_ds,  batch_size=BS, shuffle=False)

    model = model_class().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    counts = Counter(y_tr.tolist())
    cw = torch.tensor([1.0 / counts[c] for c in range(N_CLASSES)],
                       dtype=torch.float32, device=DEVICE)
    cw = cw / cw.sum() * N_CLASSES

    clf_loss = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    reg_loss = nn.HuberLoss(delta=0.5)
    opt      = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_val_acc  = 0.0
    best_epoch    = 0
    best_ckpt     = OUT_DIR / f"{model_name}_best.pt"
    no_improve    = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    print(f"Training {n_epochs} epochs  batch_size={BS}")
    for ep in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_loader, opt, clf_loss, reg_loss,
                                       gram_vals_map, alpha=0.3, device=DEVICE)
        val_acc, val_f1, val_mae, val_r2, _, _ = evaluate(model, val_loader,
                                                           gram_vals_map, device=DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = ep
            no_improve   = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_improve += 1

        if ep % 10 == 0 or ep == 1 or no_improve == 0:
            print(f"  Ep {ep:4d}/{n_epochs}  "
                  f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
                  f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  best={best_val_acc:.4f}")

        if no_improve >= patience:
            print(f"  Early stop at epoch {ep} (no improvement for {patience} epochs)")
            break

    # Final evaluation
    print(f"\nBest val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    te_acc, te_f1, te_mae, te_r2, te_pred, te_true = evaluate(
        model, te_loader, gram_vals_map, device=DEVICE
    )
    print(f"TEST: acc={te_acc:.4f}  f1={te_f1:.4f}  MAE={te_mae:.4f}g  R²={te_r2:.4f}")

    results = {
        "model_name": model_name,
        "n_params": n_params,
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "test_acc": float(te_acc),
        "test_f1": float(te_f1),
        "test_mae": float(te_mae),
        "test_r2": float(te_r2),
        "history": history,
        "predictions": {"pred": te_pred, "true": te_true},
    }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Load splits
    print("\n[1/4] Loading splits...")
    with open(SPLITS_DIR / "train.json") as f:
        train_meta_full = json.load(f)
    with open(SPLITS_DIR / "val.json") as f:
        val_meta = json.load(f)
    with open(SPLITS_DIR / "test.json") as f:
        test_meta = json.load(f)

    rng = np.random.default_rng(42)
    labels_arr = np.array([m["label"] for m in train_meta_full])
    selected = []
    for cls in range(N_CLASSES):
        idx = np.where(labels_arr == cls)[0]
        n_keep = max(1, int(len(idx) * TRAIN_FRAC))
        chosen = rng.choice(idx, size=n_keep, replace=False)
        selected.extend(chosen.tolist())
    train_meta = [train_meta_full[i] for i in sorted(selected)]

    print(f"  Train: {len(train_meta)}  Val: {len(val_meta)}  Test: {len(test_meta)}")

    # Memory-map IQ files
    print("\n[2/4] Memory-mapping IQ files...")
    iqs = {}
    for label, path in IQ_FILES.items():
        iqs[label] = np.memmap(str(path), dtype="complex64", mode="r")

    # Feature extraction
    print("\n[3/4] Feature extraction...")
    cache_dir = OUT_DIR / "feature_cache"
    cache_dir.mkdir(exist_ok=True)

    def load_or_extract(meta, split_name):
        cache_X = cache_dir / f"{split_name}_X.npy"
        cache_y = cache_dir / f"{split_name}_y.npy"
        if cache_X.exists() and cache_y.exists():
            X = np.load(cache_X)
            y = np.load(cache_y)
            print(f"  [{split_name}] Loaded from cache: {X.shape}")
            return X, y
        print(f"  Extracting {len(meta)} [{split_name}]...")
        X, y = extract_features_parallel(meta, iqs, desc=split_name)
        np.save(cache_X, X)
        np.save(cache_y, y)
        return X, y

    X_tr,  y_tr  = load_or_extract(train_meta, "train")
    X_val, y_val = load_or_extract(val_meta,   "val")
    X_te,  y_te  = load_or_extract(test_meta,  "test")

    X_tr  = np.nan_to_num(X_tr,  nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_te  = np.nan_to_num(X_te,  nan=0.0, posinf=0.0, neginf=0.0)

    # Preprocessing
    print("\n[4/4] Preprocessing...")
    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s  = scaler.transform(X_te)

    pca = PCA(n_components=PCA_DIM, random_state=42)
    X_tr_16  = pca.fit_transform(X_tr_s)
    X_val_16 = pca.transform(X_val_s)
    X_te_16  = pca.transform(X_te_s)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA {PCA_DIM}: {var_explained:.1%} variance explained")

    # Train all models
    print("\nTraining all models...")
    all_results = {}

    models_to_train = [
        ("Baseline_MLP_VQC", BaselineHybridQNN),
        ("ResNet_VQC", ResNetHybridQNN),
        ("MobileNet_VQC", MobileNetHybridQNN),
        ("EfficientNet_VQC", EfficientNetHybridQNN),
    ]

    for model_name, model_class in models_to_train:
        results = train_model(
            model_name, model_class,
            X_tr_16, y_tr, X_val_16, y_val, X_te_16, y_te,
            GRAM_VALUES, n_epochs=100, patience=30
        )
        all_results[model_name] = results

    # Save results
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    for name, res in all_results.items():
        print(f"{name:25s}  val_acc={res['best_val_acc']:.4f}  "
              f"test_acc={res['test_acc']:.4f}  test_f1={res['test_f1']:.4f}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": len(train_meta),
        "val_samples": len(val_meta),
        "test_samples": len(test_meta),
        "pca_dim": PCA_DIM,
        "pca_var": float(var_explained),
        "results": all_results,
    }

    with open(OUT_DIR / "all_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUT_DIR}/all_results.json")
    print("Done.")


if __name__ == "__main__":
    main()
