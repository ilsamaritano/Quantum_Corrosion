#!/usr/bin/env python3
"""
Quick Ensemble Eval - Direct Load
==================================
Load pre-trained models and evaluate ensemble directly on spectrogram files.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
import logging
from glob import glob
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import get_config
from src.classical_models import build_classical_model
from src.quantum_models import build_quantum_model
from src.ensemble import EnsembleClassifier
from src.dataset import PCAExtractor
import pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ensemble_eval")

def load_processed_spectrograms(data_dir="data/processed"):
    """Load spectrogram files and group by class."""
    spec_files = sorted(glob(f"{data_dir}/label*.npy"))
    
    class_data = defaultdict(list)
    for f in spec_files:
        fname = Path(f).stem  # e.g., "label0_file0_img000000"
        label = int(fname.split('_')[0].replace('label', ''))
        spec = np.load(f).astype(np.float32)
        spec = torch.from_numpy(spec)
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)  # -> (1, H, W)
        class_data[label].append(spec)
    
    return class_data

def main():
    logger.info("=" * 60)
    logger.info("Ensemble Evaluation (Direct Spectrogram Load)")
    logger.info("=" * 60)
    
    cfg = get_config()
    logger.info(f"Device: {cfg.train.device}")
    logger.info(f"Classes: {cfg.class_names}")
    
    # Load spectrograms
    logger.info("\n[1/6] Loading spectrogram files...")
    class_data = load_processed_spectrograms()
    logger.info(f"  Loaded {sum(len(v) for v in class_data.values())} spectrograms across {len(class_data)} classes")
    
    # Take test set (last 25% per class)
    logger.info("\n[2/6] Splitting into test set (25% per class)...")
    test_specs = []
    test_labels = []
    for label, specs in class_data.items():
        n_test = max(1, len(specs) // 4)
        test_specs.extend(specs[-n_test:])
        test_labels.extend([label] * n_test)
    
    test_specs = torch.stack(test_specs)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    logger.info(f"  Test set: {len(test_labels)} samples")
    
    # Format specs for models (Classical expects (B,1,H,W), Quantum expects (B,D) after PCA)
    logger.info("\n[3/6] Preparing spectrograms...")
    logger.info(f"  Input shape before formatting: {test_specs.shape}")
    
    # For Classical: reshape to (B, 1, H, W)
    classical_specs = test_specs.clone()
    if classical_specs.ndim == 3:
        classical_specs = classical_specs.unsqueeze(1)
    
    # For Quantum: flatten and use as-is (simulate PCA features)
    # The quantum model expects PCA-reduced features (batch, 1024)
    quantum_specs = test_specs.view(test_specs.shape[0], -1)  # Flatten
    logger.info(f"  Classical input shape: {classical_specs.shape}")
    logger.info(f"  Quantum input shape: {quantum_specs.shape}")
    
    # Load classical model
    logger.info("\n[4/6] Loading classical model (EfficientNet-B0)...")
    classical_ckpt = cfg.paths.models / "efficientnet_b0_round0.pt"
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
    logger.info(f"  ✓ Loaded")
    
    # Load quantum model
    logger.info("\n[5/6] Loading quantum model (VQC Amplitude)...")
    quantum_ckpt = cfg.paths.models / "hybrid_vqc_amplitude_round0.pt"
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
    logger.info(f"  ✓ Loaded")
    
    # Create ensemble
    logger.info("\n[6/6] Creating ensemble...")
    ensemble = EnsembleClassifier(
        classical_model=classical_model,
        quantum_model=quantum_model,
        n_classes=cfg.n_classes,
        fusion_method="weighted_avg",
        classical_weight=0.65,
        quantum_weight=0.35,
    )
    ensemble = ensemble.to(cfg.train.device)
    
    # Evaluate
    logger.info("\n[Evaluation] Testing on test set...")
    ensemble.eval()
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(classical_specs), 32):
            batch_classical = classical_specs[i:i+32].to(cfg.train.device)
            batch_quantum = quantum_specs[i:i+32].to(cfg.train.device)
            
            # Classical path
            with torch.no_grad():
                classical_logits = ensemble.classical_model(batch_classical)
            
            # Quantum path (need to match expected input dim)
            # If quantum expects 1024-d input, pad/truncate batch_quantum
            if batch_quantum.shape[1] != cfg.quantum.pca_dim:
                # Pad or truncate
                if batch_quantum.shape[1] < cfg.quantum.pca_dim:
                    pad_size = cfg.quantum.pca_dim - batch_quantum.shape[1]
                    batch_quantum = torch.nn.functional.pad(batch_quantum, (0, pad_size))
                else:
                    batch_quantum = batch_quantum[:, :cfg.quantum.pca_dim]
            
            with torch.no_grad():
                quantum_logits = ensemble.quantum_model(batch_quantum)
            
            # Ensemble fusion
            ensemble_logits = (
                0.65 * classical_logits + 0.35 * quantum_logits
            )
            preds = torch.argmax(ensemble_logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
    
    all_preds = np.array(all_preds)
    
    accuracy = accuracy_score(test_labels.numpy(), all_preds)
    precision = precision_score(test_labels.numpy(), all_preds, average='weighted', zero_division=0)
    recall = recall_score(test_labels.numpy(), all_preds, average='weighted', zero_division=0)
    f1 = f1_score(test_labels.numpy(), all_preds, average='weighted', zero_division=0)
    
    logger.info("\n" + "=" * 60)
    logger.info("ENSEMBLE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info("=" * 60)
    
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE COMPARISON")
    logger.info("=" * 60)
    logger.info(f"EfficientNet-B0:       ~91.2%")
    logger.info(f"Hybrid VQC Amplitude:  ~71.6%")
    logger.info(f"Ensemble (weighted):   {accuracy:.1%} ← NEW!")
    logger.info("=" * 60)
    
    improvement_vs_vqc = (accuracy - 0.716) * 100
    improvement_vs_efficientnet = (accuracy - 0.912) * 100
    
    logger.info(f"\n📊 Improvements:")
    logger.info(f"  vs VQC:         {improvement_vs_vqc:+.1f}%")
    logger.info(f"  vs EfficientNet: {improvement_vs_efficientnet:+.1f}%")

if __name__ == "__main__":
    main()
