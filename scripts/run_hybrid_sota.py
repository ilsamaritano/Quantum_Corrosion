#!/usr/bin/env python3
import json
import logging
import time
from pathlib import Path

import pennylane as qml
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

import sys
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import get_config
from src.dataset import SpectrogramDataset, get_train_transform, get_val_transform
from src.training import train_model, set_seed
from src.evaluation import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger("hybrid_sota")

class SOTA_HybridQNN(nn.Module):
    def __init__(self, n_classes, n_qubits=8, n_layers=3, backend="default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        
        # ResNet18 as classical feature extractor
        self.cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # We replace the final FC layer
        in_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(in_ftrs, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, n_qubits),
            nn.LayerNorm(n_qubits),
            nn.Tanh()  # Tanh maps inputs to [-1, 1], perfect for AngleEmbedding
        )

        dev = qml.device(backend, wires=n_qubits)
        diff_meth = "backprop" if backend == "default.qubit" else "adjoint"

        @qml.qnode(dev, interface="torch", diff_method=diff_meth)
        def qnode(inputs, weights):
            qml.AngleEmbedding(torch.pi * inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.q_weights = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 3, dtype=torch.float32))
        self.qnode = qnode
        self.head = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        # We need to replicate grayscale to 3 channels for ResNet
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.cnn(x)
        # Batched calculation via PennyLane
        q_out = self.qnode(features, self.q_weights)
        if isinstance(q_out, (tuple, list)):
            q_out = torch.stack(q_out, dim=-1)
        return self.head(q_out.to(x.dtype))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    
    out_dir = Path("results/hybrid_sota")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(cfg.paths.data_splits / "train.json") as f: train_meta = json.load(f)
    with open(cfg.paths.data_splits / "val.json") as f: val_meta = json.load(f)
    with open(cfg.paths.data_splits / "test.json") as f: test_meta = json.load(f)
    
    # We will use the full training dataset
    train_ds = SpectrogramDataset(train_meta, transform=get_train_transform(224), return_flat=False)
    val_ds = SpectrogramDataset(val_meta, transform=get_val_transform(), return_flat=False)
    test_ds = SpectrogramDataset(test_meta, transform=get_val_transform(), return_flat=False)
    
    batch_size = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    logger.info(f"Building SOTA Hybrid ResNet18-QNN (8 Qubits)... (Epochs={args.epochs}, LR={args.lr})")
    model = SOTA_HybridQNN(cfg.n_classes, n_qubits=8, n_layers=3, backend="default.qubit").to(device)
    
    t0 = time.time()
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr, # fine-tuning ResNet18 needs a small LR
        device=device,
        save_dir=out_dir,
        model_name="hybrid_sota",
        round_id=0,
        scheduler_name="cosine",
        patience=args.patience,
        use_amp=False
    )
    elapsed = time.time() - t0
    
    logger.info("Loading best checkpoint and evaluating on test set...")
    state = torch.load(out_dir / "hybrid_sota_round0.pt", map_location=device)
    model.load_state_dict(state.get("state_dict", state))
    test_res = evaluate_model(model, test_loader, cfg.class_names, device, "hybrid_sota_test", 0)
    
    logger.info("============= SOTA HYBRID QNN RESULT =============")
    logger.info(f"Test Accuracy: {test_res.accuracy*100:.2f}%")
    logger.info(f"Test F1:       {test_res.f1_score*100:.2f}%")
    logger.info(f"Training Time: {elapsed/60:.2f} min")
    logger.info("==================================================")
    with open(out_dir / "sota_summary.json", "w") as f:
        json.dump({"accuracy": test_res.accuracy, "f1": test_res.f1_score}, f, indent=2)

if __name__ == "__main__":
    main()
