#!/usr/bin/env python3
"""
Ablation Experiments on PMLB Adult Dataset
===========================================
Trains three MyMethod variants for 6 epochs and saves metrics.
"""

import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from pmlb import fetch_data
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Configuration
SEED = 42
DATASET = "adult"
MAX_ROWS = 5000
EPOCHS = 6
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
N_QUBITS = 4
N_Q_LAYERS = 2
RESULTS_DIR = Path("results/pmlb_people_comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cpu")
print(f"Device: {device}")

def maybe_cap_rows(data: pd.DataFrame, cap: int = MAX_ROWS, seed: int = SEED) -> pd.DataFrame:
    if len(data) <= cap:
        return data.copy()
    capped, _ = train_test_split(data, train_size=cap, random_state=seed, stratify=data['target'])
    return capped.reset_index(drop=True)

def split_and_encode(data: pd.DataFrame, seed: int = SEED):
    if 'target' not in data.columns:
        raise ValueError("PMLB dataset must contain a target column")

    train_df, temp_df = train_test_split(data, test_size=0.30, random_state=seed, stratify=data['target'])
    valid_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df['target'])

    feature_cols = [c for c in data.columns if c != 'target']
    categorical_cols = train_df[feature_cols].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    encoder = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )

    X_train = encoder.fit_transform(train_df[feature_cols])
    X_valid = encoder.transform(valid_df[feature_cols])
    X_test = encoder.transform(test_df[feature_cols])

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['target'].astype(str))
    y_valid = label_encoder.transform(valid_df['target'].astype(str))
    y_test = label_encoder.transform(test_df['target'].astype(str))

    X_train = np.asarray(X_train, dtype=np.float32)
    X_valid = np.asarray(X_valid, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    return {
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'encoder': encoder,
        'label_encoder': label_encoder,
        'X_train': X_train,
        'X_valid': X_valid,
        'X_test': X_test,
        'y_train': y_train,
        'y_valid': y_valid,
        'y_test': y_test,
    }

print(f"Loading dataset {DATASET}")
raw_df = fetch_data(DATASET, dropna=True)
raw_df = maybe_cap_rows(raw_df, cap=MAX_ROWS)
prep = split_and_encode(raw_df, seed=SEED)

N_FEATURES = prep['X_train'].shape[1]
N_CLASSES = len(prep['label_encoder'].classes_)
CLASS_NAMES = prep['label_encoder'].classes_.tolist()

print('Encoded feature count:', N_FEATURES)

# Quantum device and qnode
n_qubits = min(N_QUBITS, N_FEATURES)
qdev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(qdev, interface='torch', diff_method='backprop')
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class MyMethodBase(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, q_dim: int, q_layers: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.q_proj = nn.Sequential(nn.Linear(64, q_dim), nn.Sigmoid())
        self.q_weights = nn.Parameter(torch.randn(q_layers, q_dim) * 0.1)

    def _run_quantum(self, q_inputs):
        q_outputs = []
        for sample in q_inputs:
            out = qnode(sample, self.q_weights)
            if isinstance(out, list):
                out = torch.stack([val if isinstance(val, torch.Tensor) else torch.as_tensor(val, dtype=sample.dtype, device=sample.device) for val in out])
            if not isinstance(out, torch.Tensor):
                out = torch.as_tensor(out, dtype=sample.dtype, device=sample.device)
            q_outputs.append(out.float())
        return torch.stack(q_outputs, dim=0)

class VariantGateFixed05(MyMethodBase):
    def __init__(self, input_dim, output_dim, q_dim, q_layers):
        super().__init__(input_dim, output_dim, q_dim, q_layers)
        self.film = nn.Sequential(nn.Linear(q_dim, 64), nn.GELU(), nn.Linear(64, 128))
        self.head = nn.Linear(64, output_dim)

    def forward(self, x):
        c_feat = self.backbone(x)
        q_inputs = self.q_proj(c_feat) * math.pi
        q_outputs = self._run_quantum(q_inputs)
        film_params = self.film(q_outputs)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        c_mod = c_feat * (1.0 + gamma) + beta
        gate = 0.5
        fused = gate * c_mod + (1.0 - gate) * c_feat
        return self.head(fused)

class VariantNoFiLM(MyMethodBase):
    def __init__(self, input_dim, output_dim, q_dim, q_layers):
        super().__init__(input_dim, output_dim, q_dim, q_layers)
        self.gate = nn.Sequential(nn.Linear(64 + q_dim, 64), nn.Sigmoid())
        self.head = nn.Linear(64, output_dim)

    def forward(self, x):
        c_feat = self.backbone(x)
        q_inputs = self.q_proj(c_feat) * math.pi
        q_outputs = self._run_quantum(q_inputs)
        gate = self.gate(torch.cat([c_feat, q_outputs], dim=1))
        fused = gate * c_feat + (1.0 - gate) * torch.zeros_like(c_feat)
        return self.head(fused)

class VariantGaussianNoise(MyMethodBase):
    def __init__(self, input_dim, output_dim, q_dim, q_layers, noise_std=0.1):
        super().__init__(input_dim, output_dim, q_dim, q_layers)
        self.noise_std = noise_std
        self.film = nn.Sequential(nn.Linear(q_dim, 64), nn.GELU(), nn.Linear(64, 128))
        self.gate = nn.Sequential(nn.Linear(64 + q_dim, 64), nn.Sigmoid())
        self.head = nn.Linear(64, output_dim)

    def forward(self, x):
        c_feat = self.backbone(x)
        q_inputs = self.q_proj(c_feat) * math.pi
        q_outputs = self._run_quantum(q_inputs)
        if self.training:
            q_outputs = q_outputs + torch.randn_like(q_outputs) * self.noise_std
        film_params = self.film(q_outputs)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        c_mod = c_feat * (1.0 + gamma) + beta
        gate = self.gate(torch.cat([c_mod, q_outputs], dim=1))
        fused = gate * c_mod + (1.0 - gate) * c_feat
        return self.head(fused)

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
    }

def evaluate_model(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred)

def train_model(model, train_loader, valid_loader, epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, variant_name=""):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    best_epoch = -1
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_true = []
        train_pred = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            train_true.extend(yb.cpu().numpy().tolist())
            train_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())

        train_loss = running_loss / len(train_loader.dataset)
        valid_true, valid_pred = evaluate_model(model, valid_loader)
        valid_metrics = compute_metrics(valid_true, valid_pred)
        history.append({'epoch': epoch, 'train_loss': train_loss, **{f'valid_{k}': v for k, v in valid_metrics.items()}})
        print(f"{variant_name} Epoch {epoch:02d}/{epochs} train_loss={train_loss:.4f} valid_f1={valid_metrics['macro_f1']:.4f}")

        if valid_metrics['macro_f1'] > best_score:
            best_score = valid_metrics['macro_f1']
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history, {'best_epoch': best_epoch, 'best_valid_macro_f1': best_score}

# Prepare dataloaders
train_ds = TensorDataset(torch.tensor(prep['X_train'], dtype=torch.float32), torch.tensor(prep['y_train'], dtype=torch.long))
valid_ds = TensorDataset(torch.tensor(prep['X_valid'], dtype=torch.float32), torch.tensor(prep['y_valid'], dtype=torch.long))
test_ds = TensorDataset(torch.tensor(prep['X_test'], dtype=torch.float32), torch.tensor(prep['y_test'], dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

variants = {
    'gate_fixed_05': VariantGateFixed05(N_FEATURES, N_CLASSES, n_qubits, N_Q_LAYERS).to(device),
    'no_film': VariantNoFiLM(N_FEATURES, N_CLASSES, n_qubits, N_Q_LAYERS).to(device),
    'gaussian_noise': VariantGaussianNoise(N_FEATURES, N_CLASSES, n_qubits, N_Q_LAYERS).to(device),
}

results = {}
for variant_name, model in variants.items():
    print('\n' + '='*60)
    print('Training', variant_name)
    print('='*60)
    trained_model, history, best_info = train_model(model, train_loader, valid_loader, epochs=EPOCHS, variant_name=variant_name)
    test_true, test_pred = evaluate_model(trained_model, test_loader)
    test_metrics = compute_metrics(test_true, test_pred)
    checkpoint_path = RESULTS_DIR / f"ablation_{variant_name}_checkpoint.pt"
    torch.save(trained_model.state_dict(), checkpoint_path)
    predictions_path = RESULTS_DIR / f"ablation_{variant_name}_predictions.csv"
    pd.DataFrame({'y_true': test_true, 'y_pred': test_pred, 'model': variant_name}).to_csv(predictions_path, index=False)
    results[variant_name] = {
        'checkpoint': str(checkpoint_path),
        'predictions': str(predictions_path),
        'best_epoch': best_info['best_epoch'],
        'val_macro_f1': best_info['best_valid_macro_f1'],
        'test_metrics': test_metrics,
        'history': history,
    }
    print('Test metrics:', test_metrics)

summary = {
    'dataset': DATASET,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'n_qubits': n_qubits,
    'n_features': N_FEATURES,
    'n_classes': N_CLASSES,
    'device': str(device),
    'seed': SEED,
    'variants': results,
}

summary_path = RESULTS_DIR / 'ablation_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print('\nABlation summary saved to', summary_path)
print(json.dumps({k: {'val_macro_f1': v['val_macro_f1'], 'checkpoint': v['checkpoint']} for k, v in results.items()}, indent=2))
