#!/usr/bin/env python3
import math, os, random, json
from pathlib import Path
import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

SEED=42
DATASET='adult'
MAX_ROWS=5000
EPOCHS=6
BATCH_SIZE=64
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
N_QUBITS=4
N_Q_LAYERS=2
RESULTS_DIR = Path('results/pmlb_people_comparison')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=SEED):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed()
device=torch.device('cpu')

def maybe_cap_rows(data, cap=MAX_ROWS, seed=SEED):
    if len(data)<=cap:
        return data.copy()
    capped,_=train_test_split(data, train_size=cap, random_state=seed, stratify=data['target'])
    return capped.reset_index(drop=True)

def split_and_encode(data, seed=SEED):
    train_df, temp_df = train_test_split(data, test_size=0.30, random_state=seed, stratify=data['target'])
    valid_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df['target'])
    feature_cols=[c for c in data.columns if c!='target']
    categorical_cols = train_df[feature_cols].select_dtypes(include=['object','category','bool']).columns.tolist()
    numeric_cols=[c for c in feature_cols if c not in categorical_cols]
    encoder = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_cols),('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)], remainder='drop', verbose_feature_names_out=False)
    X_train=encoder.fit_transform(train_df[feature_cols])
    X_valid=encoder.transform(valid_df[feature_cols])
    X_test=encoder.transform(test_df[feature_cols])
    label_encoder=LabelEncoder()
    y_train=label_encoder.fit_transform(train_df['target'].astype(str))
    y_valid=label_encoder.transform(valid_df['target'].astype(str))
    y_test=label_encoder.transform(test_df['target'].astype(str))
    return {'X_train': np.asarray(X_train,dtype=np.float32),'X_valid':np.asarray(X_valid,dtype=np.float32),'X_test':np.asarray(X_test,dtype=np.float32),'y_train':y_train,'y_valid':y_valid,'y_test':y_test}

raw = fetch_data(DATASET, dropna=True)
raw = maybe_cap_rows(raw)
prep = split_and_encode(raw)
N_FEATURES = prep['X_train'].shape[1]
n_qubits = min(N_QUBITS, N_FEATURES)

qdev = qml.device('default.qubit', wires=n_qubits)
@qml.qnode(qdev, interface='torch', diff_method='backprop')
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class VariantGaussianNoise(nn.Module):
    def __init__(self, input_dim, output_dim, q_dim, q_layers, noise_std=0.1):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim,128), nn.ReLU(), nn.Dropout(0.15), nn.Linear(128,64), nn.ReLU())
        self.q_proj = nn.Sequential(nn.Linear(64, q_dim), nn.Sigmoid())
        self.q_weights = nn.Parameter(torch.randn(q_layers, q_dim) * 0.1)
        self.noise_std = noise_std
        self.film = nn.Sequential(nn.Linear(q_dim,64), nn.GELU(), nn.Linear(64,128))
        self.gate = nn.Sequential(nn.Linear(64 + q_dim, 64), nn.Sigmoid())
        self.head = nn.Linear(64, output_dim)

    def _run_quantum(self, q_inputs):
        q_outputs=[]
        for sample in q_inputs:
            out = qnode(sample, self.q_weights)
            if isinstance(out, list):
                out = torch.stack([val if isinstance(val, torch.Tensor) else torch.as_tensor(val, dtype=sample.dtype, device=sample.device) for val in out])
            if not isinstance(out, torch.Tensor):
                out = torch.as_tensor(out, dtype=sample.dtype, device=sample.device)
            q_outputs.append(out.float())
        return torch.stack(q_outputs, dim=0)

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
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    return {'accuracy': float(accuracy_score(y_true,y_pred)),'macro_f1': float(f1_score(y_true,y_pred, average='macro', zero_division=0)), 'macro_precision': float(precision_score(y_true,y_pred, average='macro', zero_division=0)), 'macro_recall': float(recall_score(y_true,y_pred, average='macro', zero_division=0))}

train_ds = TensorDataset(torch.tensor(prep['X_train'],dtype=torch.float32), torch.tensor(prep['y_train'],dtype=torch.long))
valid_ds = TensorDataset(torch.tensor(prep['X_valid'],dtype=torch.float32), torch.tensor(prep['y_valid'],dtype=torch.long))
test_ds = TensorDataset(torch.tensor(prep['X_test'],dtype=torch.float32), torch.tensor(prep['y_test'],dtype=torch.long))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

model = VariantGaussianNoise(N_FEATURES, len(np.unique(prep['y_train'])), n_qubits, N_Q_LAYERS).to(device)
opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
crit = nn.CrossEntropyLoss()
best_state = None
best_f1 = -1.0
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss=0.0
    for xb,yb in train_loader:
        xb=xb.to(device); yb=yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits,yb)
        loss.backward()
        opt.step()
        running_loss += loss.item()*xb.size(0)
    valid_true=[]; valid_pred=[]
    model.eval()
    with torch.no_grad():
        for xb,yb in valid_loader:
            xb=xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits,dim=1)
            valid_true.extend(yb.numpy().tolist()); valid_pred.extend(preds.cpu().numpy().tolist())
    from sklearn.metrics import f1_score
    f1 = f1_score(valid_true, valid_pred, average='macro', zero_division=0)
    print(f"gaussian_noise Epoch {epoch}/{EPOCHS} valid_f1={f1:.4f}")
    if f1>best_f1:
        best_f1=f1; best_state = model.state_dict()

model.load_state_dict(best_state)
test_true=[]; test_pred=[]
model.eval()
with torch.no_grad():
    for xb,yb in test_loader:
        logits = model(xb)
        preds = torch.argmax(logits,dim=1)
        test_true.extend(yb.numpy().tolist()); test_pred.extend(preds.cpu().numpy().tolist())
metrics = compute_metrics(test_true, test_pred)
torch.save(model.state_dict(), RESULTS_DIR / 'ablation_gaussian_noise_checkpoint.pt')
pd.DataFrame({'y_true': test_true, 'y_pred': test_pred, 'model':'gaussian_noise'}).to_csv(RESULTS_DIR / 'ablation_gaussian_noise_predictions.csv', index=False)
with open(RESULTS_DIR / 'ablation_gaussian_noise.json','w') as f:
    json.dump({'val_macro_f1': best_f1, 'test_metrics': metrics}, f, indent=2)
print('Gaussian noise done, test metrics:', metrics)
