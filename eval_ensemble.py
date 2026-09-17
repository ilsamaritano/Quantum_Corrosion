#!/usr/bin/env python3
"""
Ensemble Evaluation Script
==========================
Load pre-trained classical and quantum models, create ensemble, evaluate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
import logging
from typing import Tuple

from src.config import get_config
from src.classical_models import build_classical_model
from src.quantum_models import build_quantum_model
from src.dataset import PCASpectrogramDataset, PCAExtractor
from torch.utils.data import DataLoader
from src.ensemble import EnsembleClassifier
import pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ensemble_eval")

def load_splits(cfg):
    """Load split metadata."""
    splits_dir = cfg.paths.data_splits
    train_meta = torch.load(splits_dir / "train_meta.pt")
    val_meta = torch.load(splits_dir / "val_meta.pt")
    test_meta = torch.load(splits_dir / "test_meta.pt")
    return train_meta, val_meta, test_meta

def main():
    logger.info("=" * 60)
    logger.info("Ensemble Evaluation")
    logger.info("=" * 60)
    
    cfg = get_config()
    logger.info(f"Device: {cfg.train.device}")
    logger.info(f"Classes: {cfg.class_names}")
    
    # Load splits
    logger.info("\n[1/5] Loading dataset splits...")
    try:
        train_meta, val_meta, test_meta = load_splits(cfg)
        logger.info(f"  Train: {len(train_meta)}, Val: {len(val_meta)}, Test: {len(test_meta)}")
    except Exception as e:
        logger.error(f"Failed to load splits: {e}")
        sys.exit(1)
    
    # Load PCA
    logger.info("\n[2/5] Loading PCA reducer...")
    pca_path = cfg.paths.models / "pca_reducer.pkl"
    if not pca_path.exists():
        logger.error(f"PCA not found at {pca_path}")
        sys.exit(1)
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    logger.info(f"  PCA loaded: {pca.n_components_} components")
    
    # Build test loader
    logger.info("\n[3/5] Building test loader...")
    test_ds = PCASpectrogramDataset(test_meta, pca, use_feature_engineering=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size,
                            shuffle=False, num_workers=cfg.train.num_workers)
    logger.info(f"  Test loader: {len(test_loader)} batches")
    
    # Load classical model
    logger.info("\n[4/5] Loading classical model (EfficientNet-B0)...")
    classical_ckpt = cfg.paths.models / "efficientnet_b0_round0.pt"
    if not classical_ckpt.exists():
        logger.error(f"Checkpoint not found: {classical_ckpt}")
        sys.exit(1)
    
    classical_model = build_classical_model(
        "efficientnet_b0",
        in_channels=cfg.classical.in_channels,
        n_classes=cfg.n_classes,
        dropout=cfg.classical.dropout,
        pretrained=False,
    )
    state = torch.load(classical_ckpt, map_location="cpu")
    classical_model.load_state_dict(state.get("state_dict", state))
    classical_model = classical_model.to(cfg.train.device).eval()
    logger.info(f"  Loaded: {classical_ckpt}")
    
    # Load quantum model
    logger.info("\n[5/5] Loading quantum model (VQC Amplitude)...")
    quantum_ckpt = cfg.paths.models / "hybrid_vqc_amplitude_round0.pt"
    if not quantum_ckpt.exists():
        logger.error(f"Checkpoint not found: {quantum_ckpt}")
        sys.exit(1)
    
    quantum_model = build_quantum_model(
        input_dim=cfg.quantum.pca_dim,
        n_classes=cfg.n_classes,
        n_qubits=cfg.quantum.n_qubits,
        n_layers=cfg.quantum.n_layers,
        encoding=cfg.quantum.encoding,
        entanglement=cfg.quantum.entanglement,
        backend=cfg.quantum.backend,
        diff_method=cfg.quantum.diff_method,
        head_dims=cfg.quantum.classical_head,
    )
    state = torch.load(quantum_ckpt, map_location="cpu")
    quantum_model.load_state_dict(state.get("state_dict", state))
    quantum_model = quantum_model.to(cfg.train.device).eval()
    logger.info(f"  Loaded: {quantum_ckpt}")
    
    # Create ensemble
    logger.info("\n[Ensemble] Creating weighted ensemble (65% classical, 35% quantum)...")
    ensemble = EnsembleClassifier(
        classical_model=classical_model,
        quantum_model=quantum_model,
        n_classes=cfg.n_classes,
        fusion_method="weighted_avg",
        classical_weight=0.65,
        quantum_weight=0.35,
    )
    ensemble = ensemble.to(cfg.train.device)
    
    # Evaluate ensemble
    logger.info("\n[Evaluation] Testing ensemble on test set...")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    ensemble.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(cfg.train.device)
            logits = ensemble(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    logger.info("\n" + "=" * 60)
    logger.info("ENSEMBLE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info("=" * 60)
    
    # Save ensemble checkpoint
    ckpt_path = cfg.paths.models / "ensemble_weighted_avg_round0.pt"
    torch.save({
        "state_dict": ensemble.state_dict(),
        "config": {
            "fusion_method": "weighted_avg",
            "classical_weight": 0.65,
            "quantum_weight": 0.35,
        }
    }, ckpt_path)
    logger.info(f"\n✓ Ensemble checkpoint saved to {ckpt_path}")
    
    # Compare with baselines
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE COMPARISON")
    logger.info("=" * 60)
    logger.info(f"EfficientNet-B0:       ~91.2% (past runs)")
    logger.info(f"Hybrid VQC Amplitude:  ~71.6% (current)")
    logger.info(f"Ensemble (weighted):   {accuracy:.1%} ← NEW!")
    logger.info("=" * 60)
    
    improvement = (accuracy - 0.716) * 100
    if improvement > 0:
        logger.info(f"\n✨ Ensemble achieved +{improvement:.1f}% improvement over VQC!")
    elif improvement < -5:
        logger.warning(f"\n⚠ Ensemble underperformed vs VQC by {-improvement:.1f}%")
    else:
        logger.info(f"\n📊 Ensemble performance: {improvement:+.1f}%")

if __name__ == "__main__":
    main()
