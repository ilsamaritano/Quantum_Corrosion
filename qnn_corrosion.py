#!/usr/bin/env python3
"""
=============================================================================
Quantum Neural Network per Classificazione/Regressione di Corrosione
=============================================================================
Rappresentazione corretta degli spettrogrammi IQ (3 canali):
  Ch 0 : Magnitudine log-scale      (potenza spettrale)
  Ch 1 : Fase normalizzata [0,1]    (informazione di fase IQ)
  Ch 2 : Derivata della fase [0,1]  (frequenza istantanea)

Il canale 1-2 (fase) è spesso PIÙ discriminativo per la corrosione
perché la ruggine cambia le proprietà elettromagnetiche del bersaglio.

Pipeline:
  3-ch spettrogramma → EfficientNet-B0 (3ch, fine-tuned) → 1280d
                     → PCA(256) → AmplitudeEmbed(8q) → VQC → probs → head

Task  : Classificazione 5 classi (0.5/1/1.5/2/2.5g) + Regressione (g)
Data  : 50% training (~1849 campioni)
=============================================================================
"""
import sys, json, time, warnings, pickle
import numpy as np
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')

BASE_DIR    = Path('/home/sammarv/quantum_corrosion')
DATA_DIR    = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results' / 'qnn_final'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FRACTION = 0.50
RANDOM_SEED    = 42
N_EPOCHS_CNN   = 200
BATCH_CNN      = 128
LR_CNN         = 1e-3
N_QUBITS       = 8
N_QLAYERS      = 4
PCA_DIM        = 256
N_EPOCHS_QNN   = 50
BATCH_QNN      = 32
LR_QNN         = 5e-4
LABEL_TO_GRAMS = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5}
N_CLASSES      = 5

np.random.seed(RANDOM_SEED)

import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error, r2_score)
import xgboost as xgb
import lightgbm as lgb
import pennylane as qml

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

# ─────────────────────────────────────────────────────────────
# 1. DATASET — tutti e 3 i canali (mag + fase + d/fase)
# ─────────────────────────────────────────────────────────────
class SpectrogramDS(Dataset):
    """
    Carica (3,224,224) uint8:
      Ch0=Magnitudine  Ch1=Fase  Ch2=Derivata fase
    Normalizza ogni canale separatamente in [-1,1].
    """
    MEAN = [0.5, 0.5, 0.5]
    STD  = [0.5, 0.5, 0.5]

    def __init__(self, items, augment=False):
        self.items = items
        ops = []
        if augment:
            ops += [
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.3),
                T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
                T.RandomErasing(p=0.15, scale=(0.02, 0.06)),
            ]
        ops.append(T.Normalize(mean=self.MEAN, std=self.STD))
        self.tf = T.Compose(ops)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it  = self.items[i]
        arr = np.load(str(BASE_DIR / it['path'])).astype(np.float32) / 255.0
        x   = torch.from_numpy(arr)       # (3, 224, 224) tutti i canali
        return self.tf(x), int(it['label'])

    @property
    def labels(self):
        return np.array([it['label'] for it in self.items])


def make_balanced_sampler(labels):
    counts = Counter(labels.tolist())
    w = np.array([1.0 / counts[l] for l in labels])
    return WeightedRandomSampler(torch.from_numpy(w).float(), len(labels), replacement=True)


def load_split(name, fraction=1.0):
    with open(DATA_DIR / 'splits' / f'{name}.json') as f:
        items = json.load(f)
    if fraction < 1.0:
        n   = int(len(items) * fraction)
        idx = np.random.RandomState(RANDOM_SEED).choice(len(items), n, replace=False)
        items = [items[i] for i in sorted(idx)]
    return items


# ─────────────────────────────────────────────────────────────
# 2. CNN — EfficientNet-B0, 3 canali, fine-tuning
# ─────────────────────────────────────────────────────────────
def build_efficientnet3ch(pretrained=True, n_classes=N_CLASSES, dropout=0.3):
    """
    EfficientNet-B0 standard (3 canali RGB).
    Pretrained ImageNet → fine-tuning su spettrogrammi 3-ch.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model   = efficientnet_b0(weights=weights)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feat, n_classes),
    )
    return model


def train_cnn(tr_items, val_items, n_epochs=N_EPOCHS_CNN):
    ds_tr  = SpectrogramDS(tr_items,  augment=True)
    ds_val = SpectrogramDS(val_items, augment=False)
    sampler = make_balanced_sampler(ds_tr.labels)

    dl_tr  = DataLoader(ds_tr,  batch_size=BATCH_CNN, sampler=sampler,
                        num_workers=8, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_CNN, shuffle=False,
                        num_workers=8, pin_memory=True)

    model   = build_efficientnet3ch(pretrained=True).to(DEVICE)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR_CNN, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    gscaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc, best_state = 0.0, None
    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_loss, ep_corr, ep_tot = 0.0, 0, 0
        for imgs, lbls in dl_tr:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss   = loss_fn(logits, lbls)
            gscaler.scale(loss).backward()
            gscaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gscaler.step(opt); gscaler.update()
            ep_loss += loss.item() * len(imgs)
            ep_corr += (logits.argmax(1) == lbls).sum().item()
            ep_tot  += len(imgs)
        sched.step()

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for imgs, lbls in dl_val:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                vc += (logits.argmax(1) == lbls).sum().item()
                vt += len(imgs)
        val_acc = vc / vt
        tr_acc  = ep_corr / ep_tot

        history.append({'epoch': epoch, 'train_acc': tr_acc, 'val_acc': val_acc})

        if val_acc > best_acc:
            best_acc  = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs}  "
                  f"train={tr_acc:.4f}  val={val_acc:.4f}  best={best_acc:.4f}", flush=True)

    model.load_state_dict(best_state)
    print(f"  Fine-tuning completato — best val_acc: {best_acc:.4f}", flush=True)
    return model, history, best_acc


@torch.no_grad()
def extract_features(items, extractor, tag=""):
    ds     = SpectrogramDS(items, augment=False)
    loader = DataLoader(ds, batch_size=BATCH_CNN, num_workers=8,
                        pin_memory=True, shuffle=False)
    feats, labels = [], []
    t0 = time.time()
    print(f"  Estrazione [{tag}]: {len(items)} campioni...", flush=True)
    for imgs, lbls in loader:
        with torch.cuda.amp.autocast():
            f = extractor(imgs.to(DEVICE)).cpu().numpy()
        feats.append(f)
        labels.extend(lbls.numpy())
    feats = np.vstack(feats)
    print(f"  Done [{tag}]: {feats.shape}  {time.time()-t0:.1f}s", flush=True)
    return feats, np.array(labels)


# ─────────────────────────────────────────────────────────────
# 3. QNN — Amplitude Encoding (8 qubit)
# ─────────────────────────────────────────────────────────────
q_dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(q_dev, interface="torch", diff_method="backprop")
def vqc(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS),
                            normalize=True, pad_with=0.0)
    for l in range(N_QLAYERS):
        for q in range(N_QUBITS):
            qml.RY(weights[l, q, 0], wires=q)
            qml.RZ(weights[l, q, 1], wires=q)
        for q in range(N_QUBITS - 1):
            qml.CNOT(wires=[q, q + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])
    return qml.probs(wires=range(N_QUBITS))


q_layer = qml.qnn.TorchLayer(vqc, {"weights": (N_QLAYERS, N_QUBITS, 2)})


class HybridQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(PCA_DIM, PCA_DIM), nn.LayerNorm(PCA_DIM),
            nn.GELU(), nn.Dropout(0.1))
        self.q   = q_layer
        self.post = nn.Sequential(
            nn.Linear(2**N_QUBITS, 128), nn.LayerNorm(128),
            nn.GELU(), nn.Dropout(0.2), nn.Linear(128, N_CLASSES))

    def _norm(self, x):
        z    = self.pre(x)
        norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp(min=1e-8)
        return z / norm

    def forward(self, x):
        z = self._norm(x)
        q_out = torch.stack([self.q(z[i]) for i in range(z.shape[0])])
        return self.post(q_out)


class HybridQNNReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(PCA_DIM, PCA_DIM), nn.LayerNorm(PCA_DIM),
            nn.GELU(), nn.Dropout(0.1))
        self.q   = q_layer
        self.post = nn.Sequential(
            nn.Linear(2**N_QUBITS, 64), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        z    = self.pre(x)
        norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp(min=1e-8)
        z    = z / norm
        q_out = torch.stack([self.q(z[i]) for i in range(z.shape[0])])
        return self.post(q_out).squeeze(-1)


def train_qnn(model, X_tr, y_tr, X_v, y_v, n_epochs, task='clf'):
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_v_t  = torch.tensor(X_v,  dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long if task=='clf' else torch.float32)
    y_v_t  = torch.tensor(y_v,  dtype=torch.long if task=='clf' else torch.float32)
    loader  = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                         batch_size=BATCH_QNN, shuffle=True)
    opt     = torch.optim.Adam(model.parameters(), lr=LR_QNN, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loss_fn = (nn.CrossEntropyLoss(label_smoothing=0.05) if task=='clf'
               else nn.HuberLoss())
    best_m, best_state = (0.0 if task=='clf' else float('inf')), None

    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(xb)
        sched.step()
        model.eval()
        with torch.no_grad():
            out_v = model(X_v_t)
            m = ((out_v.argmax(1)==y_v_t).float().mean().item() if task=='clf'
                 else float(torch.abs(out_v - y_v_t).mean()))
        imp = (m > best_m) if task=='clf' else (m < best_m)
        if imp:
            best_m     = m
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            lbl = 'acc' if task=='clf' else 'MAE'
            print(f"  Epoch {epoch:3d}/{n_epochs}  loss={ep_loss/len(X_tr):.4f}  "
                  f"val_{lbl}={m:.4f}  best={best_m:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model, best_m


def predict_clf(model, X):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(X_t), BATCH_QNN):
            out.extend(model(X_t[i:i+BATCH_QNN]).argmax(1).numpy())
    return np.array(out)


def predict_reg(model, X):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(X_t), BATCH_QNN):
            out.extend(model(X_t[i:i+BATCH_QNN]).numpy())
    return np.array(out)


# ─────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 68, flush=True)
    print("QNN Corrosione v5 — 3-ch IQ Spectrogram + Fine-Tune + Amplitude QNN", flush=True)
    print(f"Training: {TRAIN_FRACTION*100:.0f}%  CNN: {N_EPOCHS_CNN}ep  "
          f"QNN: {N_QUBITS}q×{N_QLAYERS}L {N_EPOCHS_QNN}ep", flush=True)
    print("=" * 68, flush=True)

    # 1. Splits
    print("\n[1/7] Splits...", flush=True)
    tr_items  = load_split('train', TRAIN_FRACTION)
    val_items = load_split('val')
    te_items  = load_split('test')
    print(f"  Train: {len(tr_items)}  Val: {len(val_items)}  Test: {len(te_items)}", flush=True)
    print(f"  Distribuzione: {Counter(it['label'] for it in tr_items)}", flush=True)

    # 2. CNN fine-tuning (3 canali)
    cnn_ckpt = RESULTS_DIR / 'efficientnet3ch_finetune.pt'
    if cnn_ckpt.exists():
        print("\n[2/7] CNN da checkpoint...", flush=True)
        model_full = build_efficientnet3ch(pretrained=False).to(DEVICE)
        model_full.load_state_dict(torch.load(cnn_ckpt, map_location='cpu'))
        model_full.eval()
        ds_val = SpectrogramDS(val_items)
        dl_val = DataLoader(ds_val, batch_size=BATCH_CNN, num_workers=8, pin_memory=True)
        vc, vt = 0, 0
        with torch.no_grad():
            for imgs, lbls in dl_val:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                vc += (model_full(imgs).argmax(1)==lbls).sum().item()
                vt += len(imgs)
        best_cnn_acc = vc / vt
        print(f"  CNN val accuracy: {best_cnn_acc:.4f}", flush=True)
    else:
        print("\n[2/7] Fine-tuning EfficientNet-B0 (3-ch: mag+fase+dφ)...", flush=True)
        model_full, cnn_hist, best_cnn_acc = train_cnn(tr_items, val_items, N_EPOCHS_CNN)
        torch.save(model_full.state_dict(), cnn_ckpt)
        print(f"  Checkpoint: {cnn_ckpt}", flush=True)

    # 3. Feature extraction
    feat_cache = RESULTS_DIR / 'cnn3ch_features.pkl'
    if feat_cache.exists():
        print("\n[3/7] Feature da cache...", flush=True)
        with open(feat_cache, 'rb') as f:
            cache = pickle.load(f)
        X_tr, y_tr = cache['X_tr'], cache['y_tr']
        X_v,  y_v  = cache['X_v'],  cache['y_v']
        X_te, y_te = cache['X_te'], cache['y_te']
        print(f"  Shape: {X_tr.shape}", flush=True)
    else:
        print("\n[3/7] Estrazione feature...", flush=True)
        extractor = nn.Sequential(*list(model_full.children())[:-1],
                                   nn.Flatten()).to(DEVICE).eval()
        # Per EfficientNet-B0: features + avgpool (classifier rimosso)
        extractor = model_full
        extractor.classifier = nn.Identity()
        for p in extractor.parameters(): p.requires_grad = False
        X_tr, y_tr = extract_features(tr_items,  extractor, "train")
        X_v,  y_v  = extract_features(val_items, extractor, "val")
        X_te, y_te = extract_features(te_items,  extractor, "test")
        with open(feat_cache, 'wb') as f:
            pickle.dump({'X_tr': X_tr, 'y_tr': y_tr,
                         'X_v':  X_v,  'y_v':  y_v,
                         'X_te': X_te, 'y_te': y_te}, f)

    del model_full
    torch.cuda.empty_cache()

    # 4. Preprocessing per QNN
    print("\n[4/7] StandardScaler + PCA...", flush=True)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_v_s  = sc.transform(X_v)
    X_te_s = sc.transform(X_te)

    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_SEED)
    X_tr_pca = pca.fit_transform(X_tr_s)
    X_v_pca  = pca.transform(X_v_s)
    X_te_pca = pca.transform(X_te_s)
    print(f"  PCA {PCA_DIM}: {pca.explained_variance_ratio_.sum()*100:.1f}% varianza", flush=True)

    qsc = StandardScaler()
    X_tr_q = qsc.fit_transform(X_tr_pca)
    X_v_q  = qsc.transform(X_v_pca)
    X_te_q = qsc.transform(X_te_pca)

    grams = lambda y: np.array([LABEL_TO_GRAMS[l] for l in y], dtype=np.float32)
    g_tr, g_v, g_te = grams(y_tr), grams(y_v), grams(y_te)
    results = {'EfficientNet3ch_val': best_cnn_acc}

    # 5. Classici su feature CNN
    print("\n[5/7] Modelli classici su feature CNN 3-ch...", flush=True)

    print("  XGBoost clf...", flush=True)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', early_stopping_rounds=30,
        random_state=RANDOM_SEED, n_jobs=-1)
    xgb_clf.fit(X_tr_s, y_tr, eval_set=[(X_v_s, y_v)], verbose=False)
    xgb_acc = accuracy_score(y_te, xgb_clf.predict(X_te_s))
    results['XGBoost_clf'] = xgb_acc
    print(f"  XGBoost → {xgb_acc:.4f} ({xgb_acc*100:.2f}%)", flush=True)

    print("  XGBoost reg...", flush=True)
    xgb_reg = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=30, random_state=RANDOM_SEED, n_jobs=-1)
    xgb_reg.fit(X_tr_s, g_tr, eval_set=[(X_v_s, g_v)], verbose=False)
    xgb_gpred = xgb_reg.predict(X_te_s)
    results['XGBoost_MAE'] = mean_absolute_error(g_te, xgb_gpred)
    results['XGBoost_R2']  = r2_score(g_te, xgb_gpred)
    print(f"  XGBoost reg → MAE={results['XGBoost_MAE']:.4f}g  R²={results['XGBoost_R2']:.4f}", flush=True)

    print("  LightGBM clf...", flush=True)
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    lgb_clf.fit(X_tr_s, y_tr, eval_set=[(X_v_s, y_v)],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(-1)])
    lgb_acc = accuracy_score(y_te, lgb_clf.predict(X_te_s))
    results['LightGBM_clf'] = lgb_acc
    print(f"  LightGBM → {lgb_acc:.4f} ({lgb_acc*100:.2f}%)", flush=True)

    # 6. QNN
    print("\n[6/7] Quantum Neural Network (Amplitude Encoding)...", flush=True)
    print(f"  {PCA_DIM}d → AmplitudeEmbed → {N_QUBITS}q VQC×{N_QLAYERS} "
          f"→ probs({2**N_QUBITS}) → head → {N_CLASSES}cl", flush=True)

    print("\n  [6a] QNN Classificatore...", flush=True)
    tq = time.time()
    qnn_clf = HybridQNN()
    qnn_clf, best_val_q = train_qnn(qnn_clf, X_tr_q, y_tr, X_v_q, y_v, N_EPOCHS_QNN)
    print(f"  Training: {time.time()-tq:.1f}s", flush=True)
    qnn_pred = predict_clf(qnn_clf, X_te_q)
    qnn_acc  = accuracy_score(y_te, qnn_pred)
    results['QNN_clf'] = qnn_acc
    print(f"  QNN → Test Accuracy: {qnn_acc:.4f} ({qnn_acc*100:.2f}%)", flush=True)

    print("\n  [6b] QNN Regressore...", flush=True)
    tq = time.time()
    qnn_reg = HybridQNNReg()
    qnn_reg, _ = train_qnn(qnn_reg, X_tr_q, g_tr, X_v_q, g_v, N_EPOCHS_QNN, task='reg')
    print(f"  Training: {time.time()-tq:.1f}s", flush=True)
    qnn_gpred = predict_reg(qnn_reg, X_te_q)
    results['QNN_MAE'] = mean_absolute_error(g_te, qnn_gpred)
    results['QNN_R2']  = r2_score(g_te, qnn_gpred)
    print(f"  QNN reg → MAE={results['QNN_MAE']:.4f}g  R²={results['QNN_R2']:.4f}", flush=True)

    # 7. Report
    grams_labels = ['0.5g', '1.0g', '1.5g', '2.0g', '2.5g']
    total_min    = (time.time() - t0) / 60

    print("\n" + "=" * 68, flush=True)
    print("[7/7] RISULTATI FINALI", flush=True)
    print("=" * 68, flush=True)
    print(f"\nDataset : train={len(tr_items)} ({TRAIN_FRACTION*100:.0f}%) | "
          f"val={len(val_items)} | test={len(te_items)}", flush=True)
    print(f"Features: EfficientNet-B0 3-ch (mag+fase+dφ) → 1280d → PCA {PCA_DIM}d", flush=True)
    print(f"QNN     : {N_QUBITS} qubit × {N_QLAYERS} strati = "
          f"{N_QLAYERS*N_QUBITS*2} param quantistici", flush=True)

    print("\n── CLASSIFICAZIONE (5 classi) ───────────────────────────────────", flush=True)
    best_name, best_acc_val = "", 0.0
    for name, key in [
        ("EfficientNet-B0 3ch (val acc)", 'EfficientNet3ch_val'),
        ("CNN+XGBoost (test)",            'XGBoost_clf'),
        ("CNN+LightGBM (test)",           'LightGBM_clf'),
        ("CNN+QNN ibrido (test)",         'QNN_clf'),
    ]:
        acc = results[key]
        star = " ★" if (acc > best_acc_val and "val" not in name) else ""
        if acc > best_acc_val and "val" not in name:
            best_acc_val = acc; best_name = name
        print(f"  {name:38s}  {acc:.4f}  ({acc*100:.2f}%){star}", flush=True)

    print("\n── REGRESSIONE (grammi) ─────────────────────────────────────────", flush=True)
    for name, mk, rk in [
        ("CNN+XGBoost", 'XGBoost_MAE', 'XGBoost_R2'),
        ("CNN+QNN",     'QNN_MAE',     'QNN_R2'),
    ]:
        print(f"  {name:38s}  MAE={results[mk]:.4f}g  R²={results[rk]:.4f}", flush=True)

    print(f"\n── Confusion Matrix QNN ─────────────────────────────────────────", flush=True)
    cm = confusion_matrix(y_te, qnn_pred)
    print("        " + "   ".join(f"{l:>5}" for l in grams_labels), flush=True)
    for i, row in enumerate(cm):
        print(f"  {grams_labels[i]:>5}  " + "   ".join(f"{v:5d}" for v in row), flush=True)

    print(f"\n── Classification Report QNN ───────────────────────────────────", flush=True)
    print(classification_report(y_te, qnn_pred, target_names=grams_labels), flush=True)

    print(f"\n── Migliore: {best_name} → {best_acc_val*100:.2f}% ──", flush=True)
    print(f"── Tempo totale: {total_min:.1f} min ─────────────────────────────", flush=True)

    # Salva
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    torch.save(qnn_clf.state_dict(), RESULTS_DIR / 'qnn_clf.pt')
    torch.save(qnn_reg.state_dict(), RESULTS_DIR / 'qnn_reg.pt')
    with open(RESULTS_DIR / 'xgb_clf.pkl', 'wb') as f: pickle.dump(xgb_clf, f)
    print(f"\n  Salvato in {RESULTS_DIR}", flush=True)
    return results


if __name__ == '__main__':
    main()
