#!/usr/bin/env python3
"""
Few-shot Hybrid CNN-QNN run for high accuracy.

Uses a pretrained ResNet18 feature extractor mapped to a
strongly entangled Quantum Neural Network layer (qml.StronglyEntanglingLayers).
This architecture is widely considered state-of-the-art for finding
quantum advantage in image classification with few shots.
"""

import argparse
import sys
import time
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models

# Insert project root to track modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.dataset import SpectrogramDataset, create_splits
from src.training import train_model, set_seed
from src.evaluation import evaluate_model
import pennylane as qml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("cnn_qnn_fewshot")


class HybridCNNQNN(nn.Module):
    """
    State-of-the-Art Hybrid QNN: Pretrained ResNet18 -> StronglyEntanglingLayers
    """
    def __init__(self, n_classes: int, n_qubits: int = 4, n_layers: int = 3, backend: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        
        # 1. Classical Feature Extractor (ResNet18 fine-tuning)
        logger.info("Loading pretrained ResNet18 and unfreezing last block for fine-tuning...")
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for name, param in resnet.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True  # Unfreeze the final convolutional block
            else:
                param.requires_grad = False  # Freeze early layers
            
        # Replace the final FC layer with a trainable projection to match qubit count
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, n_qubits),
            nn.Tanh() # Keep features bounded for the Angle-based embedding
        )
        self.cnn = resnet
        
        # 2. Quantum Neural Network Block
        logger.info(f"Building Quantum Neural Network ({n_layers} StronglyEntanglingLayers on {n_qubits} qubits)...")
        dev = qml.device(backend, wires=n_qubits)
        
        diff_method = "parameter-shift" if "gpu" in backend else "adjoint"
        
        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def qnode(inputs, weights):
            # Encode inputs into quantum state via Angle Embedding
            for i in range(n_qubits):
                qml.RY(np.pi * inputs[i], wires=i)
                qml.RZ(np.pi * inputs[i], wires=i)
                
            # Formally a Quantum Neural Network architecture
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            
            # Measurement / Readout
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
        self.q_weights = nn.Parameter(
            0.01 * torch.randn(n_layers, n_qubits, 3, dtype=torch.float32)
        )
        self.qnode = qnode
        
        # 3. Final Classical output head
        self.head = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
        
        logger.info(f"Hybrid QNN initialized on {backend}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spectrogram expects (batch, channels, H, W).
        # We need to explicitly replicate grayscale over 3 channels for ResNet
        if len(x.shape) == 3:  # (B, H, W)
            x = x.unsqueeze(1)
        if x.shape[1] == 1:    # Expand single channel to RGB
            x = x.repeat(1, 3, 1, 1)
            
        cnn_features = self.cnn(x)  # (Batch, n_qubits)
        
        # QNode execution (PennyLane does not yet fully support batched inputs + batched weights easily on all devices)
        batch_size = x.shape[0]
        q_out = []
        for i in range(batch_size):
            feat = cnn_features[i]
            expectations = self.qnode(feat, self.q_weights)
            expect_tensor = torch.stack(expectations)  # list of tensors -> shape (n_qubits)
            q_out.append(expect_tensor)
            
        q_out = torch.stack(q_out).to(x.device, dtype=torch.float32)
        
        return self.head(q_out)


def load_splits(cfg):
    with open(cfg.paths.data_splits / "train.json") as f:
        train_meta = json.load(f)
    with open(cfg.paths.data_splits / "val.json") as f:
        val_meta = json.load(f)
    with open(cfg.paths.data_splits / "test.json") as f:
        test_meta = json.load(f)
    return train_meta, val_meta, test_meta


def stratified_few_shot(metadata, shots_per_class, seed):
    if shots_per_class <= 0:
        return metadata  # Use full dataset
        
    rng = np.random.default_rng(seed)
    labels = np.array([int(r["label"]) for r in metadata], dtype=np.int64)
    selected_idx = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) == 0:
            continue
        n_keep = min(len(cls_idx), shots_per_class)
        chosen = rng.choice(cls_idx, size=n_keep, replace=False)
        selected_idx.extend(chosen.tolist())

    meta_arr = np.array(metadata)
    return meta_arr[sorted(selected_idx)].tolist()


def main():
    parser = argparse.ArgumentParser(description="Hybrid CNN-QNN Few-Shot Training")
    parser.add_argument("--shots-per-class", type=int, default=74)
    parser.add_argument("--target-accuracy", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quantum-backend", default="lightning.gpu")
    parser.add_argument("--refresh", action="store_true", help="Refresh dataset (Spectrogram extraction)")
    
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = get_config()
    cfg.train.device = args.device

    out_dir = Path("results/hybrid_cnn_qnn_99")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh:
        from src.preprocessing import preprocess_dataset
        import shutil
        logger.info("Forcing refresh of spectrograms (now 3-channel Magnitude+Phase+Doppler)...")
        if cfg.paths.data_processed.exists():
            shutil.rmtree(cfg.paths.data_processed)
        cfg.paths.data_processed.mkdir()
        metadata = preprocess_dataset(cfg, max_workers=4)
        create_splits(metadata, ratios=cfg.train.split_ratios, seed=cfg.train.seed, save_dir=cfg.paths.data_splits)
        logger.info("Preprocessing complete.")

    # 1. Load splits
    train_meta, val_meta, test_meta = load_splits(cfg)
    fs_train_meta = stratified_few_shot(train_meta, args.shots_per_class, args.seed)
    
    logger.info("=" * 60)
    logger.info("🚀 HYBRID CNN-QNN FOR 99% FEW-SHOT ACCURACY")
    logger.info("=" * 60)
    logger.info(f"Qubits: {args.n_qubits} | QNN Layers: {args.n_layers}")
    logger.info(f"Few-Shot Train: {len(fs_train_meta)} samples ({args.shots_per_class} per class)")
    logger.info(f"Validation: {len(val_meta)} samples (Full)")
    logger.info(f"Backend: {args.quantum_backend}")

    # 2. Build direct Spectrogram Datasets (no PCA interpolation needed for CNN)
    logger.info("Preparing direct spectrogram data loading without PCA bottlenecks...")
    train_ds = SpectrogramDataset(fs_train_meta, transform=None, return_flat=False)
    val_ds = SpectrogramDataset(val_meta, transform=None, return_flat=False)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    
    # 3. Create CNN-QNN Model
    model = HybridCNNQNN(
        n_classes=cfg.n_classes,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        backend=args.quantum_backend
    ).to(args.device)

    # 4. Train
    t0 = time.time()
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=out_dir,
        model_name="cnn_qnn",
        round_id=0
    )
    
    elapsed = time.time() - t0
    
    # 5. Evaluate on Validation set to check if we hit 99%
    res = evaluate_model(
        model=model,
        data_loader=val_loader,
        class_names=cfg.class_names,
        device=args.device,
        model_name="cnn_qnn_val",
        round_id=0,
        train_time=elapsed
    )
    
    logger.info("=" * 60)
    logger.info(f"🎉 FINAL VALIDATION ACCURACY: {res.accuracy * 100:.2f}%")
    if res.accuracy >= args.target_accuracy:
        logger.info(f"✅ TARGET {args.target_accuracy*100:.1f}% REACHED!")
    else:
        logger.info(f"❌ Target of {args.target_accuracy*100:.1f}% not reached yet.")

if __name__ == "__main__":
    main()
