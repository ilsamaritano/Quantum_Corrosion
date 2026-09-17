#!/usr/bin/env python3
"""Quick-run wrapper for corrosion_merging pipeline (SKIP_QNAS=True)
"""
import os, sys, time, json, re, warnings, random, copy
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
from torch.utils.data import DataLoader, TensorDataset

import pennylane as qml
from pennylane import numpy as pnp
import optuna

warnings.filterwarnings("ignore")

# Allow quick runs without Optuna
SKIP_QNAS = True

# ── Strict Reproducibility ────────────────────────────────────────────────────
SEED = 42
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)

# ── Paths & 9-Class Configuration ─────────────────────────────────────────────
BASE_DIR = Path("/home/sammarv/quantum_corrosion")
SPLITS_DIR = BASE_DIR / "data/splits"
OUT_DIR = BASE_DIR / "results/qnn_v7_9class_qnas"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IQ_FILES = {
    0: BASE_DIR / "data/raw/0.5/0.5.iq",
    1: BASE_DIR / "data/raw/1/1.iq",
    2: BASE_DIR / "data/raw/1.5/1.5.iq",
    3: BASE_DIR / "data/raw/2/2.iq",
    4: BASE_DIR / "data/raw/2.5/2.5.iq",
    5: BASE_DIR / "stack/1/1gSTACK",
    6: BASE_DIR / "stack/1.5/1.5gSTACK",
    7: BASE_DIR / "stack/2/2gSTACK",
    8: BASE_DIR / "stack/2.5/2.5gSTACK",
}

GRAM_VALUES = {0:0.5,1:1.0,2:1.5,3:2.0,4:2.5,5:1.0,6:1.5,7:2.0,8:2.5}
LABEL_NAMES = [
    "spread_0.5g", "spread_1.0g", "spread_1.5g", "spread_2.0g", "spread_2.5g",
    "stack_1.0g", "stack_1.5g", "stack_2.0g", "stack_2.5g"
]
N_CLASSES = 9
TRAIN_FRAC = 1.00
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WORKERS = 8

FFT_SIZE = 4096
N_STACKS = 16
OVERLAP = 0.25
HOP = int(FFT_SIZE * (1 - OVERLAP))
SAMPLES_PER_IMG = N_STACKS * HOP + (FFT_SIZE - HOP)
WIN = get_window("hann", FFT_SIZE).astype(np.float32)

print("Starting quick-run: SKIP_QNAS=", SKIP_QNAS)

# feature extraction functions
def _get_img_idx(path: str) -> int:
    m = re.search(r"img(\d+)", str(path))
    return int(m.group(1)) if m else 0

def extract_one(iq: np.memmap, img_idx: int) -> np.ndarray:
    start = img_idx * SAMPLES_PER_IMG
    end   = start + SAMPLES_PER_IMG
    if end > len(iq): return None
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
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * psd_mean).sum() / (psd_mean.sum() + 1e-12)))
    log_psd = np.log(psd_mean + 1e-12)
    flatness = float(np.exp(log_psd.mean()) / (psd_mean.mean() + 1e-12))
    p = psd_mean / (psd_mean.sum() + 1e-12)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    frame_power = mag.sum(axis=1)
    power_var   = float(frame_power.var())
    power_skew  = float(skew(frame_power))
    power_kurt  = float(scipy_kurtosis(frame_power))
    stats = np.array([total_power, centroid, bandwidth, flatness, entropy, power_var, power_skew, power_kurt], dtype=np.float32)
    phase = np.angle(spec).astype(np.float32)
    phase_mean_std  = float(np.std(phase.mean(axis=0)))
    phase_var_mean  = float(phase.var(axis=1).mean())
    inst_freq = np.diff(np.unwrap(phase, axis=0), axis=0)
    if_mean = float(inst_freq.mean())
    if_std  = float(inst_freq.std())
    i_pwr = (frames.real ** 2).mean()
    q_pwr = (frames.imag ** 2).mean()
    iq_imbalance = float(i_pwr / (q_pwr + 1e-12))
    iq_corr = float(np.corrcoef(frames.real.flatten()[:2048], frames.imag.flatten()[:2048])[0, 1])
    amp = np.abs(frames.flatten()[:4096])
    amp_kurt = float(scipy_kurtosis(amp))
    amp_skew = float(skew(amp))
    phase_feats = np.array([phase_mean_std, phase_var_mean, if_mean, if_std, iq_imbalance, iq_corr, amp_kurt, amp_skew], dtype=np.float32)
    return np.concatenate([psd_512, var_512, stats, phase_feats])

def extract_features_parallel(meta_list, iqs, desc="", n_workers=N_WORKERS):
    n = len(meta_list)
    feats, labels = [None] * n, [None] * n
    t0 = time.time()
    def _worker(i, rec):
        label = int(rec["label"])
        idx   = _get_img_idx(rec["path"])
        f = extract_one(iqs[label], idx)
        return i, f, label
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker, i, rec): i for i, rec in enumerate(meta_list)}
        done = 0
        for fut in as_completed(futs):
            i, f, lbl = fut.result()
            feats[i], labels[i] = f, lbl
            done += 1
            if done % 200 == 0 or done == n:
                elapsed = time.time() - t0
                print(f"  {done}/{n}  elapsed={elapsed:.0f}s  ETA={elapsed/done*(n-done):.0f}s")
    valid =[(f, l) for f, l in zip(feats, labels) if f is not None]
    if not valid:
        return np.empty((0, 1040), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack([v[0] for v in valid]), np.array([v[1] for v in valid])

# Models
class PriorGuidedQuantumFiLM(nn.Module):
    def __init__(self, c_dim=64, q_dim=4):
        super().__init__()
        self.film_net = nn.Sequential(nn.Linear(q_dim, 32), nn.SiLU(), nn.LayerNorm(32), nn.Linear(32, c_dim*3))
        self.gate_logits = nn.Sequential(nn.Linear(c_dim + q_dim, c_dim), nn.LayerNorm(c_dim))
        self.q_proj = nn.Sequential(nn.Linear(q_dim, 16), nn.SiLU())
        self.classifier = nn.Sequential(nn.Linear(c_dim + 16, 64), nn.SiLU(), nn.Dropout(0.3), nn.Linear(64, N_CLASSES))
        self.regressor = nn.Sequential(nn.Linear(c_dim + 16, 32), nn.SiLU(), nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(self, c, q):
        film_params = self.film_net(q)
        gamma, beta, alpha = torch.chunk(film_params, 3, dim=1)
        c_mod = c * (gamma + 1.0) + beta
        raw_gate = self.gate_logits(torch.cat([c_mod, q], dim=1))
        gate_val = torch.sigmoid(raw_gate + alpha)
        fused_c = (gate_val * c_mod) + ((1 - gate_val) * c)
        q_res = self.q_proj(q)
        final_features = torch.cat([fused_c, q_res], dim=1)
        return self.classifier(final_features), self.regressor(final_features)

class DenseHybridQNN(nn.Module):
    def __init__(self, n_qubits, n_layers, ansatz, angle_scaling, dropout_c):
        super().__init__()
        self.angle_scaling = angle_scaling
        self.classical_backbone = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_c), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_c))
        self.qnn_proj = nn.Sequential(nn.Linear(256, n_qubits), nn.Sigmoid())
        dev = qml.device("default.qubit", wires=n_qubits)
        @qml.qnode(dev, interface='torch')
        def _qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            if ansatz == "StronglyEntangling":
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            else:
                qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        if ansatz == "StronglyEntangling":
            w_shape = {"weights": (n_layers, n_qubits, 3)}
        else:
            w_shape = {"weights": (n_layers, n_qubits)}
        self.qnn = qml.qnn.TorchLayer(_qnode, w_shape)
        self.fusion_module = PriorGuidedQuantumFiLM(c_dim=64, q_dim=n_qubits)
    def forward(self, x):
        c_feats = self.classical_backbone(x)
        q_in = self.qnn_proj(x) * torch.tensor(self.angle_scaling, device=x.device)
        q_feats = self.qnn(q_in)
        logits, grams = self.fusion_module(c_feats, q_feats)
        return logits, grams.squeeze(1)

# Training utilities

def train_epoch(model, loader, opt, clf_loss_fn, reg_loss_fn, alpha=0.3):
    model.train()
    total_loss = correct = n = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        g_b = torch.tensor([GRAM_VALUES[int(l)] for l in y_b.cpu()], dtype=torch.float32, device=DEVICE)
        opt.zero_grad()
        logits, grams = model(X_b)
        loss = (1 - alpha) * clf_loss_fn(logits, y_b) + alpha * reg_loss_fn(grams, g_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * len(y_b)
        correct    += (logits.argmax(1) == y_b).sum().item()
        n          += len(y_b)
    return total_loss / max(1, n), correct / max(1, n)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_pred, all_true, all_gram_pred, all_gram_true = [], [], [],[]
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        logits, grams = model(X_b)
        all_pred.extend(logits.argmax(1).cpu().tolist())
        all_true.extend(y_b.cpu().tolist())
        all_gram_pred.extend(grams.cpu().tolist())
        all_gram_true.extend([GRAM_VALUES[int(l)] for l in y_b.cpu()])
    if len(all_true) == 0:
        return 0.0, 0.0, 0.0, 0.0, [], []
    acc  = accuracy_score(all_true, all_pred)
    f1   = f1_score(all_true, all_pred, average="macro")
    mae  = mean_absolute_error(all_gram_true, all_gram_pred)
    r2   = r2_score(all_gram_true, all_gram_pred)
    return acc, f1, mae, r2, all_pred, all_true

# Main

def main():
    print("[1/5] Loading splits & creating memmaps...")
    with open(SPLITS_DIR / "train.json") as f: train_meta_full = json.load(f)
    with open(SPLITS_DIR / "val.json") as f: val_meta = json.load(f)
    with open(SPLITS_DIR / "test.json") as f: test_meta = json.load(f)
    rng = np.random.default_rng(SEED)
    labels_arr = np.array([int(m["label"]) for m in train_meta_full])
    existing_iq_files = {label: path for label, path in IQ_FILES.items() if path.exists()}
    available_labels = [label for label in sorted(existing_iq_files) if label in set(labels_arr.tolist())]
    if not available_labels:
        raise RuntimeError("No overlapping labels found between the train split and available IQ files.")
    selected = []
    for cls in available_labels:
        idx = np.where(labels_arr == cls)[0]
        if len(idx) == 0:
            continue
        n_keep = max(1, int(len(idx) * TRAIN_FRAC))
        selected.extend(rng.choice(idx, size=n_keep, replace=False).tolist())
    train_meta = [train_meta_full[i] for i in sorted(selected)]
    iqs = {l: np.memmap(str(p), dtype="complex64", mode="r") for l, p in existing_iq_files.items()}
    print("[2/5] Extracting features...")
    cache_dir = OUT_DIR / "feature_cache"
    cache_dir.mkdir(exist_ok=True)
    def load_or_extract(meta, split_name):
        cX, cy = cache_dir / f"{split_name}_X.npy", cache_dir / f"{split_name}_y.npy"
        if cX.exists() and cy.exists(): return np.load(cX), np.load(cy)
        X, y = extract_features_parallel(meta, iqs, desc=split_name)
        np.save(cX, X); np.save(cy, y)
        return X, y
    X_tr, y_tr = load_or_extract(train_meta, "train")
    X_val, y_val = load_or_extract(val_meta, "val")
    X_te, y_te = load_or_extract(test_meta, "test")
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    print("[3/5] Preprocessing")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)
    pca = PCA(n_components=256, random_state=SEED)
    X_tr_pca = pca.fit_transform(X_tr_s)
    X_val_pca = pca.transform(X_val_s)
    X_te_pca = pca.transform(X_te_s)
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    T_X_tr, T_y_tr = T(X_tr_pca), torch.tensor(y_tr, dtype=torch.long)
    T_X_val, T_y_val = T(X_val_pca), torch.tensor(y_val, dtype=torch.long)
    T_X_te, T_y_te = T(X_te_pca), torch.tensor(y_te, dtype=torch.long)
    counts = Counter(y_tr.tolist())
    cw_list = [1.0 / counts.get(c, 1) for c in range(N_CLASSES)]
    cw = torch.tensor(cw_list, dtype=torch.float32, device=DEVICE)
    cw = cw / cw.sum() * N_CLASSES
    print("[4/5] Optuna or defaults")
    def objective(trial):
        n_qubits = trial.suggest_categorical("n_qubits", [4, 6, 8, 10])
        n_layers = trial.suggest_int("n_layers", 1, 4)
        ansatz = trial.suggest_categorical("ansatz", ["StronglyEntangling", "BasicEntangling"])
        scale_name = trial.suggest_categorical("angle_scaling", ["pi_half", "pi", "two_pi"])
        angle_scaling = np.pi/2.0 if scale_name == "pi_half" else (np.pi if scale_name == "pi" else 2.0*np.pi)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout_c = trial.suggest_float("dropout_c", 0.1, 0.5)
        batch_size = trial.suggest_categorical("batch_size",[16, 32])
        tr_loader = DataLoader(TensorDataset(T_X_tr, T_y_tr), batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(TensorDataset(T_X_val, T_y_val), batch_size=batch_size, shuffle=False)
        model = DenseHybridQNN(n_qubits, n_layers, ansatz, angle_scaling, dropout_c).to(DEVICE)
        clf_loss = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
        reg_loss = nn.HuberLoss(delta=0.5)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        for ep in range(25):
            train_epoch(model, tr_loader, opt, clf_loss, reg_loss, alpha=0.3)
            acc, f1, mae, r2, _, _ = evaluate(model, val_loader)
            trial.report(f1, ep)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        return f1
    if SKIP_QNAS:
        best_params = {"n_qubits":6, "n_layers":2, "ansatz":"BasicEntangling", "angle_scaling":"pi", "lr":1e-3, "weight_decay":1e-5, "dropout_c":0.2, "batch_size":32}
        print("Using defaults:", best_params)
    else:
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
        study.optimize(objective, n_trials=35)
        best_params = study.best_trial.params
    print("[5/5] Final training & evaluation")
    best_bs = best_params["batch_size"]
    tr_loader = DataLoader(TensorDataset(T_X_tr, T_y_tr), batch_size=best_bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(T_X_val, T_y_val), batch_size=best_bs, shuffle=False)
    te_loader = DataLoader(TensorDataset(T_X_te, T_y_te), batch_size=best_bs, shuffle=False)
    angle_scaling_map = {"pi_half": np.pi/2, "pi": np.pi, "two_pi": 2*np.pi}
    final_model = DenseHybridQNN(n_qubits=best_params["n_qubits"], n_layers=best_params["n_layers"], ansatz=best_params["ansatz"], angle_scaling=angle_scaling_map[best_params["angle_scaling"]], dropout_c=best_params["dropout_c"]).to(DEVICE)
    clf_loss = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    reg_loss = nn.HuberLoss(delta=0.5)
    opt = torch.optim.AdamW(final_model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    best_val_f1 = -1.0
    best_weights = copy.deepcopy(final_model.state_dict())
    patience = 20
    no_improve = 0
    for ep in range(1, 101):
        tr_loss, tr_acc = train_epoch(final_model, tr_loader, opt, clf_loss, reg_loss, alpha=0.3)
        val_acc, val_f1, val_mae, val_r2, _, _ = evaluate(final_model, val_loader)
        scheduler.step(val_f1)
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            best_weights = copy.deepcopy(final_model.state_dict())
            torch.save(best_weights, OUT_DIR / "final_qnn_9class_best.pt")
        else:
            no_improve += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:3d}/100 | Tr Acc: {tr_acc:.4f} | Val F1: {val_f1:.4f} | Val MAE: {val_mae:.4f} | Best F1: {best_val_f1:.4f}")
        if no_improve >= patience:
            print(f"  Early stopping at epoch {ep}")
            break
    print("\nFINAL EVALUATION ON TEST SET")
    if best_weights is None:
        print("No best weights saved — aborting evaluation")
        return
    final_model.load_state_dict(best_weights)
    te_acc, te_f1, te_mae, te_r2, te_pred, te_true = evaluate(final_model, te_loader)
    print(f"TEST ACC: {te_acc:.4f} | F1: {te_f1:.4f} | MAE: {te_mae:.4f} | R2: {te_r2:.4f}")
    print("CLASS REPORT:\n", classification_report(te_true, te_pred, labels=list(range(N_CLASSES)), target_names=LABEL_NAMES, digits=4, zero_division=0))
    print("CONFUSION MATRIX:\n", confusion_matrix(te_true, te_pred, labels=list(range(N_CLASSES))))

if __name__ == '__main__':
    main()
