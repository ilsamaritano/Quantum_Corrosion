#!/usr/bin/env python3
import os, math, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

# Config
SEED = 42
DATASET = 'iris'
# Use a large cap so the full dataset is used by default
MAX_ROWS = 10**6
EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
N_QUBITS = 4
N_Q_LAYERS = 2
RESULTS_DIR = Path('results/pmlb_iris_comparison')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# seed
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# device (match notebook behavior)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Load dataset
print('Fetching', DATASET)
raw = fetch_data(DATASET, dropna=True)
if len(raw) > MAX_ROWS:
    raw = raw.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)

# Preprocess: simple StandardScaler for numeric features
feature_cols = [c for c in raw.columns if c!='target']
X = raw[feature_cols]
y = raw['target'].astype(str)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in numeric_cols]

if len(cat_cols)>0:
    # One-hot encode categoricals if any (iris has none)
    from sklearn.preprocessing import OneHotEncoder
    enc = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_cols), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)], remainder='drop', verbose_feature_names_out=False)
else:
    from sklearn.preprocessing import StandardScaler
    enc = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_cols)], remainder='drop', verbose_feature_names_out=False)

X_enc = enc.fit_transform(X)
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(X_enc, y_enc, test_size=0.3, random_state=SEED, stratify=y_enc)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

X_train = X_train.astype(np.float32); X_valid = X_valid.astype(np.float32); X_test = X_test.astype(np.float32)

n_features = X_train.shape[1]
n_classes = len(le.classes_)
print('n_features', n_features, 'n_classes', n_classes)

# Quantum device
n_qubits = min(N_QUBITS, n_features)
qdev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(qdev, interface='torch', diff_method='backprop')
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Models
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim,64), nn.ReLU(), nn.Linear(64,output_dim))
    def forward(self,x):
        return self.net(x)

class QNNLinearConcat(nn.Module):
    def __init__(self, input_dim, output_dim, q_dim, q_layers):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim,64), nn.ReLU(), nn.Linear(64,q_dim), nn.Sigmoid())
        self.q_weights = nn.Parameter(torch.randn(q_layers, q_dim)*0.1)
        self.classical_head = nn.Sequential(nn.Linear(input_dim,64), nn.ReLU())
        self.fusion = nn.Linear(64+q_dim, output_dim)
    def forward(self,x):
        q_inputs = self.encoder(x)*math.pi
        q_outputs = []
        for sample in q_inputs:
            out = qnode(sample, self.q_weights)
            if isinstance(out,list):
                out = torch.stack([v if isinstance(v,torch.Tensor) else torch.as_tensor(v, dtype=sample.dtype, device=sample.device) for v in out])
            if not isinstance(out, torch.Tensor):
                out = torch.as_tensor(out, dtype=sample.dtype, device=sample.device)
            q_outputs.append(out.float())
        q_outputs = torch.stack(q_outputs, dim=0)
        c_out = self.classical_head(x)
        fused = torch.cat([c_out, q_outputs], dim=1)
        return self.fusion(fused)

class MyMethod(nn.Module):
    def __init__(self, input_dim, output_dim, q_dim, q_layers):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU())
        self.q_proj = nn.Sequential(nn.Linear(64, q_dim), nn.Sigmoid())
        self.q_weights = nn.Parameter(torch.randn(q_layers, q_dim)*0.1)
        self.film = nn.Sequential(nn.Linear(q_dim,32), nn.GELU(), nn.Linear(32, 128))
        self.gate = nn.Sequential(nn.Linear(64+q_dim, 64), nn.Sigmoid())
        self.head = nn.Linear(64, output_dim)
    def forward(self,x):
        c_feat = self.backbone(x)
        q_inputs = self.q_proj(c_feat)*math.pi
        q_outputs = []
        for sample in q_inputs:
            out = qnode(sample, self.q_weights)
            if isinstance(out,list):
                out = torch.stack([v if isinstance(v,torch.Tensor) else torch.as_tensor(v, dtype=sample.dtype, device=sample.device) for v in out])
            if not isinstance(out, torch.Tensor):
                out = torch.as_tensor(out, dtype=sample.dtype, device=sample.device)
            q_outputs.append(out.float())
        q_outputs = torch.stack(q_outputs, dim=0)
        film_params = self.film(q_outputs)
        # split
        if film_params.shape[1] % 2 == 0:
            gamma, beta = torch.chunk(film_params, 2, dim=1)
        else:
            # pad
            pad = torch.zeros(film_params.shape[0], 1, device=film_params.device, dtype=film_params.dtype)
            film_params = torch.cat([film_params, pad], dim=1)
            gamma, beta = torch.chunk(film_params, 2, dim=1)
        c_mod = c_feat * (1.0 + gamma) + beta
        gate = self.gate(torch.cat([c_mod, q_outputs], dim=1))
        fused = gate * c_mod + (1.0 - gate) * c_feat
        return self.head(fused)

# Prepare dataloaders
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
valid_ds = TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# Training util
def compute_metrics(y_true, y_pred):
    return {'accuracy': float(accuracy_score(y_true,y_pred)), 'macro_f1': float(f1_score(y_true,y_pred, average='macro', zero_division=0))}

def evaluate_model(model, loader):
    model.eval()
    y_true=[]; y_pred=[]
    with torch.no_grad():
        for xb,yb in loader:
            xb = xb.float().to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(yb.numpy().tolist()); y_pred.extend(preds.cpu().numpy().tolist())
    return y_true, y_pred

# Instantiate models
baseline = BaselineMLP(n_features, n_classes).to(device)
qnn = QNNLinearConcat(n_features, n_classes, n_qubits, N_Q_LAYERS).to(device)
my = MyMethod(n_features, n_classes, n_qubits, N_Q_LAYERS).to(device)

models = {'baseline': baseline, 'qnn': qnn, 'my': my}

results = {}
for name, model in models.items():
    print('--- Training', name)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    best_f1 = -1; best_state = None
    for epoch in range(1, EPOCHS+1):
        model.train(); running=0.0
        for xb,yb in train_loader:
            xb=xb.float().to(device); yb=yb.long().to(device)
            opt.zero_grad(); logits = model(xb)
            loss = crit(logits, yb); loss.backward(); opt.step(); running += loss.item()*xb.size(0)
        yv, pv = evaluate_model(model, valid_loader)
        metrics = compute_metrics(yv,pv)
        print(f"{name} epoch {epoch} valid_f1={metrics['macro_f1']:.4f}")
        if metrics['macro_f1']>best_f1:
            best_f1=metrics['macro_f1']; best_state = model.state_dict()
    model.load_state_dict(best_state)
    yt, pt = evaluate_model(model, test_loader)
    tm = compute_metrics(yt,pt)
    results[name] = {'val_best_macro_f1': best_f1, 'test_metrics': tm}
    torch.save(model.state_dict(), RESULTS_DIR / f'{name}.pt')

# Save metrics
with open(RESULTS_DIR / 'metrics.json','w') as f:
    json.dump({'dataset': DATASET, 'results': results}, f, indent=2)
print('Saved metrics to', RESULTS_DIR / 'metrics.json')
print(json.dumps(results, indent=2))
