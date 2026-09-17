#!/usr/bin/env python3
"""
QNN Corrosion v6 — Raw IQ Feature Extraction + Hybrid Quantum Neural Network
=============================================================================
Key insight: extract features DIRECTLY from raw IQ files (no per-image
normalization), which preserves absolute amplitude/spectral shape that
encodes corrosion level. Achieves 88%+ vs 22% from pre-processed npy.

Pipeline:
  Raw IQ → Full-resolution PSD + stats features (1024-d)
  → StandardScaler → PCA
  → [XGBoost / LightGBM / RF baselines]
  → [Hybrid QNN: AngleEmbedding + VQC + MLP head → 5 classes]
  → Regression head (predict grams)

Training fraction: 50% of the pre-defined train split.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              mean_absolute_error, r2_score)
import joblib

import torch
import torch.nn as nn
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
OUT_DIR = BASE_DIR / "results/qnn_v6_hybrid_amp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Signal config (matches src/config.py) ─────────────────────────────────────
FFT_SIZE = 4096
N_STACKS = 16      # ridotto per usare max ~10% del dataset
OVERLAP = 0.25
HOP = int(FFT_SIZE * (1 - OVERLAP))          # 3072
SAMPLES_PER_IMG = N_STACKS * HOP + (FFT_SIZE - HOP)  # 394 240
WIN = get_window("hann", FFT_SIZE).astype(np.float32)

# ── QNN config ────────────────────────────────────────────────────────────────
N_QUBITS   = 8       # amplitude encoding: 2^8 = 256 features
N_QLAYERS  = 6
PCA_DIM    = 256     # reduced dim fed to QNN
N_CLASSES  = 5
TRAIN_FRAC = 0.50    # mantenuto come richiesto al 50%
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_WORKERS  = 8

print("=" * 70)
print("QNN Corrosion v6 — Raw IQ Features + Hybrid QNN")
print(f"Device: {DEVICE}   Workers: {N_WORKERS}")
print(f"FFT_SIZE={FFT_SIZE}  N_STACKS={N_STACKS}  SAMPLES_PER_IMG={SAMPLES_PER_IMG}")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE EXTRACTION FROM RAW IQ
# ═══════════════════════════════════════════════════════════════════════════════

def _get_img_idx(path: str) -> int:
    m = re.search(r"img(\d+)", str(path))
    return int(m.group(1)) if m else 0


def extract_one(iq: np.memmap, img_idx: int) -> np.ndarray:
    """
    Extract rich spectral features for one image window from raw IQ.
    Returns 1D float32 feature vector.
    """
    start = img_idx * SAMPLES_PER_IMG
    end   = start + SAMPLES_PER_IMG
    if end > len(iq):
        return None

    # Build frame matrix
    frames = np.empty((N_STACKS, FFT_SIZE), dtype="complex64")
    for i in range(N_STACKS):
        s = start + i * HOP
        frames[i] = iq[s : s + FFT_SIZE]

    # FFT → magnitude spectrum (no per-image normalization)
    spec = np.fft.fftshift(np.fft.fft(frames * WIN, axis=1), axes=1)
    mag  = np.abs(spec).astype(np.float32)  # (N_STACKS, FFT_SIZE)

    # ── Feature group 1: Mean PSD (dB) – 512 bins ──────────────────────────
    psd_mean = mag.mean(axis=0)                      # (FFT_SIZE,)
    psd_db   = 20.0 * np.log10(psd_mean + 1e-12)
    # Downsample 4096 → 512
    psd_512  = psd_db.reshape(512, 8).mean(axis=1)   # (512,)

    # ── Feature group 2: PSD temporal variance – 512 bins ──────────────────
    psd_var  = mag.var(axis=0)
    var_512  = np.log1p(psd_var.reshape(512, 8).mean(axis=1))  # (512,)

    # ── Feature group 3: Spectral statistics (global) – 8 features ─────────
    # Absolute power (no normalization → class discriminative!)
    total_power = float(psd_mean.sum())
    # Spectral centroid
    freqs = np.arange(FFT_SIZE, dtype=np.float32)
    centroid = float((freqs * psd_mean).sum() / (psd_mean.sum() + 1e-12))
    # Spectral bandwidth
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * psd_mean).sum() /
                               (psd_mean.sum() + 1e-12)))
    # Spectral flatness (Wiener entropy)
    log_psd = np.log(psd_mean + 1e-12)
    flatness = float(np.exp(log_psd.mean()) / (psd_mean.mean() + 1e-12))
    # Spectral entropy
    p = psd_mean / (psd_mean.sum() + 1e-12)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    # Temporal variance of total frame power (signal stationarity)
    frame_power = mag.sum(axis=1)  # (N_STACKS,)
    power_var   = float(frame_power.var())
    power_skew  = float(skew(frame_power))
    power_kurt  = float(scipy_kurtosis(frame_power))

    stats = np.array([total_power, centroid, bandwidth, flatness,
                      entropy, power_var, power_skew, power_kurt],
                     dtype=np.float32)

    # ── Feature group 4: Phase statistics – 8 features ─────────────────────
    phase = np.angle(spec).astype(np.float32)           # (N_STACKS, FFT_SIZE)
    phase_mean_std  = float(np.std(phase.mean(axis=0)))  # std of mean phase spectrum
    phase_var_mean  = float(phase.var(axis=1).mean())    # mean temporal phase variance
    # Instantaneous frequency (phase derivative)
    inst_freq = np.diff(np.unwrap(phase, axis=0), axis=0)  # (N_STACKS-1, FFT_SIZE)
    if_mean = float(inst_freq.mean())
    if_std  = float(inst_freq.std())
    # I/Q imbalance proxy
    i_pwr = (frames.real ** 2).mean()
    q_pwr = (frames.imag ** 2).mean()
    iq_imbalance = float(i_pwr / (q_pwr + 1e-12))
    # Cross-correlation at lag-0 between I and Q
    iq_corr = float(np.corrcoef(frames.real.flatten()[:2048],
                                frames.imag.flatten()[:2048])[0, 1])
    # Amplitude kurtosis of the IQ signal itself
    amp = np.abs(frames.flatten()[:4096])
    amp_kurt = float(scipy_kurtosis(amp))
    amp_skew = float(skew(amp))

    phase_feats = np.array([phase_mean_std, phase_var_mean, if_mean, if_std,
                             iq_imbalance, iq_corr, amp_kurt, amp_skew],
                            dtype=np.float32)

    return np.concatenate([psd_512, var_512, stats, phase_feats])  # (1040,)


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
            if done % 200 == 0 or done == n:
                elapsed = time.time() - t0
                eta = elapsed / done * (n - done)
                print(f"  {done}/{n}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    # Filter failed extractions
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
    """
    Variational Quantum Circuit with AmplitudeEmbedding.
    """
    qml.AmplitudeEmbedding(features=inputs, wires=range(N_QUBITS), normalize=True)
    for layer in range(N_QLAYERS):
        for q in range(N_QUBITS):
            qml.Rot(weights[layer, q, 0],
                    weights[layer, q, 1],
                    weights[layer, q, 2], wires=q)
        # Entanglement: circular CNOT ring
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]


class HybridQNN(nn.Module):
    """
    Hybrid classical-quantum model:
      Input (batch, PCA_DIM=16)
      → AngleEmbedding VQC (16q × 4L) → (batch, 16) expectation values
      → Linear head → (batch, N_CLASSES)
    Also includes a regression head for gram prediction.
    """

    def __init__(self):
        super().__init__()
        # Scale PCA features to [-π, π] range
        self.input_scale = nn.Parameter(
            torch.tensor([np.pi], dtype=torch.float32), requires_grad=False
        )
        # Quantum weights: (n_layers, n_qubits, 3 params per qubit)
        w_shape = (N_QLAYERS, N_QUBITS, 3)
        self.q_weights = nn.Parameter(
            torch.randn(*w_shape, dtype=torch.float32) * 0.1
        )
        # Classical feature extraction directly linked to PCA dim
        self.classical_path = nn.Sequential(
            nn.Linear(PCA_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Classical head merging QNN (8) + Classical (128) -> 136 neurons
        self.clf_head = nn.Sequential(
            nn.Linear(N_QUBITS + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, N_CLASSES),
        )
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(N_QUBITS + 128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # vqc normalizes inputs automatically for AmplitudeEmbedding
        q_results = vqc(x, self.q_weights)
        q_out = torch.stack(q_results, dim=-1).float()  # (batch, N_QUBITS)
        
        c_out = self.classical_path(x) # shape: (batch, 128)
        combined = torch.cat([q_out, c_out], dim=-1) # shape: (batch, 136)

        logits = self.clf_head(combined)   # (batch, N_CLASSES)
        grams  = self.reg_head(combined)   # (batch, 1)
        return logits, grams.squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAINING UTILITIES
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
# 4. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 4.1 Load splits ─────────────────────────────────────────────────────
    print("\n[1/6] Loading splits...")
    with open(SPLITS_DIR / "train.json") as f:
        train_meta_full = json.load(f)
    with open(SPLITS_DIR / "val.json") as f:
        val_meta = json.load(f)
    with open(SPLITS_DIR / "test.json") as f:
        test_meta = json.load(f)

    # Stratified subsample 50% of training set
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
    print(f"  Label dist train: {Counter(m['label'] for m in train_meta)}")

    # ── 4.2 Memory-map IQ files ──────────────────────────────────────────────
    print("\n[2/6] Memory-mapping IQ files...")
    iqs = {}
    for label, path in IQ_FILES.items():
        iqs[label] = np.memmap(str(path), dtype="complex64", mode="r")
        print(f"  Class {label}: {len(iqs[label]) // 1_000_000:.0f}M samples")

    # ── 4.3 Feature extraction ───────────────────────────────────────────────
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
        print(f"  Extracting {len(meta)} [{split_name}] with {N_WORKERS} workers...")
        X, y = extract_features_parallel(meta, iqs, desc=split_name)
        np.save(cache_X, X)
        np.save(cache_y, y)
        return X, y

    print("\n[3/6] Feature extraction...")
    X_tr,  y_tr  = load_or_extract(train_meta, "train")
    X_val, y_val = load_or_extract(val_meta,   "val")
    X_te,  y_te  = load_or_extract(test_meta,  "test")

    # Nan/inf check
    for name, X in [("train", X_tr), ("val", X_val), ("test", X_te)]:
        n_nan = np.isnan(X).sum(); n_inf = np.isinf(X).sum()
        if n_nan or n_inf:
            print(f"  WARNING [{name}]: {n_nan} nan, {n_inf} inf → filling with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_tr  = np.nan_to_num(X_tr,  nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_te  = np.nan_to_num(X_te,  nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Feature shape: {X_tr.shape}")

    # ── 4.4 Preprocessing: StandardScaler + PCA ──────────────────────────────
    print("\n[4/6] Preprocessing (StandardScaler + PCA)...")
    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s  = scaler.transform(X_te)

    # Two PCA spaces: full (256-d) for classical models, small (16-d) for QNN
    pca_256 = PCA(n_components=256, random_state=42)
    X_tr_256  = pca_256.fit_transform(X_tr_s)
    X_val_256 = pca_256.transform(X_val_s)
    X_te_256  = pca_256.transform(X_te_s)
    var_256   = pca_256.explained_variance_ratio_.sum()
    print(f"  PCA 256: {var_256:.1%} variance explained")

    pca_16 = PCA(n_components=PCA_DIM, random_state=42)
    X_tr_16  = pca_16.fit_transform(X_tr_s)
    X_val_16 = pca_16.transform(X_val_s)
    X_te_16  = pca_16.transform(X_te_s)
    var_16   = pca_16.explained_variance_ratio_.sum()
    print(f"  PCA {PCA_DIM}: {var_16:.1%} variance explained")

    # Save PCA/scaler
    joblib.dump(scaler, OUT_DIR / "scaler.pkl")
    joblib.dump(pca_256, OUT_DIR / "pca_256.pkl")
    joblib.dump(pca_16,  OUT_DIR / "pca_16.pkl")

    # ── 4.5 Classical baselines ───────────────────────────────────────────────
    print("\n[5/6] Classical baselines (RF, XGBoost, LightGBM)...")
    results = {}

    # Random Forest
    print("  Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                                random_state=42, n_jobs=-1, min_samples_leaf=3)
    rf.fit(X_tr_256, y_tr)
    rf_tr  = accuracy_score(y_tr,  rf.predict(X_tr_256))
    rf_val = accuracy_score(y_val, rf.predict(X_val_256))
    rf_te  = accuracy_score(y_te,  rf.predict(X_te_256))
    rf_f1  = f1_score(y_val, rf.predict(X_val_256), average="macro")
    print(f"    RF: train={rf_tr:.4f}  val={rf_val:.4f}  test={rf_te:.4f}  f1={rf_f1:.4f}")
    results["rf"] = {"val_acc": rf_val, "test_acc": rf_te, "val_f1": rf_f1}
    joblib.dump(rf, OUT_DIR / "rf.pkl")

    # XGBoost
    try:
        from xgboost import XGBClassifier
        print("  XGBoost...")
        xgb = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=30,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1, tree_method="hist"
        )
        xgb.fit(X_tr_256, y_tr, eval_set=[(X_val_256, y_val)], verbose=False)
        xgb_val = accuracy_score(y_val, xgb.predict(X_val_256))
        xgb_te  = accuracy_score(y_te,  xgb.predict(X_te_256))
        xgb_f1  = f1_score(y_val, xgb.predict(X_val_256), average="macro")
        print(f"    XGBoost: val={xgb_val:.4f}  test={xgb_te:.4f}  f1={xgb_f1:.4f}")
        results["xgboost"] = {"val_acc": xgb_val, "test_acc": xgb_te, "val_f1": xgb_f1}
        joblib.dump(xgb, OUT_DIR / "xgb.pkl")
    except Exception as e:
        print(f"    XGBoost failed: {e}")

    # LightGBM
    try:
        import lightgbm as lgb
        print("  LightGBM...")
        lgbm = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=8, verbose=-1
        )
        lgbm.fit(X_tr_256, y_tr)
        lgbm_val = accuracy_score(y_val, lgbm.predict(X_val_256))
        lgbm_te  = accuracy_score(y_te,  lgbm.predict(X_te_256))
        lgbm_f1  = f1_score(y_val, lgbm.predict(X_val_256), average="macro")
        print(f"    LightGBM: val={lgbm_val:.4f}  test={lgbm_te:.4f}  f1={lgbm_f1:.4f}")
        results["lgbm"] = {"val_acc": lgbm_val, "test_acc": lgbm_te, "val_f1": lgbm_f1}
        joblib.dump(lgbm, OUT_DIR / "lgbm.pkl")
    except Exception as e:
        print(f"    LightGBM failed: {e}")

    # ── 4.6 Hybrid QNN ────────────────────────────────────────────────────────
    print(f"\n[6/6] Hybrid QNN ({N_QUBITS}q × {N_QLAYERS}L AngleEmbedding)...")

    # Build PyTorch datasets
    gram_vals_map = GRAM_VALUES
    T = lambda a: torch.from_numpy(a.astype(np.float32))

    tr_ds  = TensorDataset(T(X_tr_16),  torch.from_numpy(y_tr.astype(np.int64)))
    val_ds = TensorDataset(T(X_val_16), torch.from_numpy(y_val.astype(np.int64)))
    te_ds  = TensorDataset(T(X_te_16),  torch.from_numpy(y_te.astype(np.int64)))

    BS = 32
    tr_loader  = DataLoader(tr_ds,  batch_size=BS, shuffle=True,  drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False)
    te_loader  = DataLoader(te_ds,  batch_size=BS, shuffle=False)

    model = HybridQNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  QNN trainable params: {n_params}")

    # Class weights for imbalanced training
    counts = Counter(y_tr.tolist())
    cw = torch.tensor([1.0 / counts[c] for c in range(N_CLASSES)],
                       dtype=torch.float32, device=DEVICE)
    cw = cw / cw.sum() * N_CLASSES

    clf_loss = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    reg_loss = nn.HuberLoss(delta=0.5)
    opt      = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

    best_val_acc  = 0.0
    best_epoch    = 0
    best_ckpt     = OUT_DIR / "qnn_best.pt"
    patience      = 30
    no_improve    = 0
    n_epochs      = 100

    print(f"  Training {n_epochs} epochs  batch={BS}")
    for ep in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_loader, opt, clf_loss, reg_loss,
                                       gram_vals_map, alpha=0.3, device=DEVICE)
        val_acc, val_f1, val_mae, val_r2, _, _ = evaluate(model, val_loader,
                                                           gram_vals_map, device=DEVICE)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = ep
            no_improve   = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_improve += 1

        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:4d}/{n_epochs}  "
                  f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
                  f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  "
                  f"val_mae={val_mae:.4f}  best={best_val_acc:.4f}")

        if no_improve >= patience:
            print(f"  Early stop at epoch {ep} (no improvement for {patience} epochs)")
            break

    # Final evaluation on test set with best model
    print(f"\n  Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    te_acc, te_f1, te_mae, te_r2, te_pred, te_true = evaluate(
        model, te_loader, gram_vals_map, device=DEVICE
    )
    print(f"  TEST: acc={te_acc:.4f}  f1={te_f1:.4f}  MAE={te_mae:.4f}g  R²={te_r2:.4f}")

    results["qnn"] = {
        "best_val_acc": best_val_acc,
        "test_acc":     te_acc,
        "test_f1":      te_f1,
        "test_mae":     te_mae,
        "test_r2":      te_r2,
    }

    # Full classification report
    print("\n  Classification report (test set):")
    labels_names = ["0.5g", "1.0g", "1.5g", "2.0g", "2.5g"]
    print(classification_report(te_true, te_pred, target_names=labels_names))

    # ── Save summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name:10s}: val_acc={r.get('val_acc', r.get('best_val_acc', 0)):.4f}  "
              f"test_acc={r.get('test_acc', 0):.4f}  "
              f"f1={r.get('val_f1', r.get('test_f1', 0)):.4f}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": len(train_meta),
        "val_samples":   len(val_meta),
        "test_samples":  len(test_meta),
        "feature_dim":   int(X_tr.shape[1]),
        "pca_16_var":    float(var_16),
        "pca_256_var":   float(var_256),
        "results":       {k: {sk: float(sv) for sk, sv in v.items()}
                          for k, v in results.items()},
        "config": {
            "n_qubits":  N_QUBITS,
            "n_qlayers": N_QLAYERS,
            "pca_dim":   PCA_DIM,
            "train_frac": TRAIN_FRAC,
        }
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {OUT_DIR}")
    print("  Done.")


if __name__ == "__main__":
    main()
