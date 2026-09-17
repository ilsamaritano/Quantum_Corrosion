#!/usr/bin/env python3
"""
Evaluate a quantum-assisted ensemble using already-trained checkpoints.

- Classical branch: spectrogram image -> CNN logits
- Quantum branch: PCA features -> VQC logits
- Fusion: weighted average of logits, weight tuned on validation set
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.classical_models import build_classical_model
from src.dataset import PCAExtractor, PCASpectrogramDataset, SpectrogramDataset, get_val_transform
from src.quantum_models import build_quantum_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("quantum_assisted_eval")


def load_splits_json(cfg) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    splits_dir = cfg.paths.data_splits
    with open(splits_dir / "train.json") as f:
        train_meta = json.load(f)
    with open(splits_dir / "val.json") as f:
        val_meta = json.load(f)
    with open(splits_dir / "test.json") as f:
        test_meta = json.load(f)
    return train_meta, val_meta, test_meta


def build_classical(arch: str, ckpt_path: Path, cfg):
    model = build_classical_model(
        arch,
        n_classes=cfg.n_classes,
        in_channels=cfg.classical.in_channels,
        pretrained=False,
        dropout=cfg.classical.dropout,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    return model


def build_quantum(ckpt_path: Path, cfg):
    # Explicit architecture matching existing trained checkpoints.
    model = build_quantum_model(
        input_dim=1024,
        n_classes=cfg.n_classes,
        n_qubits=8,
        n_layers=10,
        encoding="amplitude",
        entanglement="full",
        backend="default.qubit",
        diff_method="backprop",
        head_dims=[512, 256, 128],
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    return model


@torch.no_grad()
def infer_logits(model: torch.nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).cpu().numpy()
        all_logits.append(logits)
        all_labels.append(y.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="weighted", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="weighted", zero_division=0)),
        "f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def tune_weight(
    val_logits_c: np.ndarray,
    val_logits_q: np.ndarray,
    val_labels: np.ndarray,
    weights: Iterable[float],
) -> Tuple[float, Dict[str, float]]:
    best_w = 0.0
    best_metrics = {"accuracy": -1.0, "precision": 0.0, "recall": 0.0, "f1": -1.0}

    for w in weights:
        fused = w * val_logits_c + (1.0 - w) * val_logits_q
        m = metrics_from_logits(fused, val_labels)
        # Prioritize F1, break ties by accuracy.
        if (m["f1"] > best_metrics["f1"]) or (
            abs(m["f1"] - best_metrics["f1"]) < 1e-12 and m["accuracy"] > best_metrics["accuracy"]
        ):
            best_w = float(w)
            best_metrics = m

    return best_w, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantum-assisted ensemble with pretrained models")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--classical-arch", default="mobilenetv2")
    parser.add_argument(
        "--classical-ckpt",
        default="results/models/best_over90_mobilenetv2_round8.pt",
    )
    parser.add_argument(
        "--quantum-ckpt",
        default="results/models/hybrid_vqc_amplitude_round1.pt",
    )
    parser.add_argument(
        "--pca-path",
        default="results/models/pca_reducer.pkl",
    )
    parser.add_argument(
        "--out-csv",
        default="results/quantum_assisted_over90_summary.csv",
    )
    args = parser.parse_args()

    cfg = get_config()

    classical_ckpt = Path(args.classical_ckpt)
    quantum_ckpt = Path(args.quantum_ckpt)
    pca_path = Path(args.pca_path)

    for p in [classical_ckpt, quantum_ckpt, pca_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    _, val_meta, test_meta = load_splits_json(cfg)

    val_img_ds = SpectrogramDataset(val_meta, transform=get_val_transform())
    test_img_ds = SpectrogramDataset(test_meta, transform=get_val_transform())

    pca = PCAExtractor(n_components=1024).load(pca_path)
    val_q_ds = PCASpectrogramDataset(val_meta, pca, use_feature_engineering=True)
    test_q_ds = PCASpectrogramDataset(test_meta, pca, use_feature_engineering=True)

    val_img_loader = DataLoader(val_img_ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    test_img_loader = DataLoader(test_img_ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    val_q_loader = DataLoader(val_q_ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    test_q_loader = DataLoader(test_q_ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    logger.info("Loading pretrained checkpoints...")
    classical_model = build_classical(args.classical_arch, classical_ckpt, cfg)
    quantum_model = build_quantum(quantum_ckpt, cfg)

    logger.info("Running validation inference...")
    val_logits_c, val_labels_c = infer_logits(classical_model, val_img_loader, args.device)
    val_logits_q, val_labels_q = infer_logits(quantum_model, val_q_loader, args.device)
    if not np.array_equal(val_labels_c, val_labels_q):
        raise RuntimeError("Validation label mismatch between classical and quantum loaders")

    logger.info("Running test inference...")
    test_logits_c, test_labels_c = infer_logits(classical_model, test_img_loader, args.device)
    test_logits_q, test_labels_q = infer_logits(quantum_model, test_q_loader, args.device)
    if not np.array_equal(test_labels_c, test_labels_q):
        raise RuntimeError("Test label mismatch between classical and quantum loaders")

    classical_val = metrics_from_logits(val_logits_c, val_labels_c)
    quantum_val = metrics_from_logits(val_logits_q, val_labels_q)
    classical_test = metrics_from_logits(test_logits_c, test_labels_c)
    quantum_test = metrics_from_logits(test_logits_q, test_labels_q)

    weight_grid = np.linspace(0.50, 1.00, 51)
    best_w, best_val = tune_weight(val_logits_c, val_logits_q, val_labels_c, weight_grid)

    fused_test_logits = best_w * test_logits_c + (1.0 - best_w) * test_logits_q
    fused_test = metrics_from_logits(fused_test_logits, test_labels_c)

    logger.info("=" * 64)
    logger.info("Validation")
    logger.info(f"Classical: Acc={classical_val['accuracy']:.4f} F1={classical_val['f1']:.4f}")
    logger.info(f"Quantum:   Acc={quantum_val['accuracy']:.4f} F1={quantum_val['f1']:.4f}")
    logger.info(f"Best fuse: w_classical={best_w:.2f} -> Acc={best_val['accuracy']:.4f} F1={best_val['f1']:.4f}")
    logger.info("-" * 64)
    logger.info("Test")
    logger.info(f"Classical: Acc={classical_test['accuracy']:.4f} F1={classical_test['f1']:.4f}")
    logger.info(f"Quantum:   Acc={quantum_test['accuracy']:.4f} F1={quantum_test['f1']:.4f}")
    logger.info(f"Fused:     Acc={fused_test['accuracy']:.4f} F1={fused_test['f1']:.4f}")
    logger.info("=" * 64)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "f1", "precision", "recall", "classical_weight"])
        writer.writerow(["classical", classical_test["accuracy"], classical_test["f1"], classical_test["precision"], classical_test["recall"], 1.0])
        writer.writerow(["quantum", quantum_test["accuracy"], quantum_test["f1"], quantum_test["precision"], quantum_test["recall"], 0.0])
        writer.writerow(["quantum_assisted_fused", fused_test["accuracy"], fused_test["f1"], fused_test["precision"], fused_test["recall"], best_w])

    print(f"Saved summary to {out_csv}")


if __name__ == "__main__":
    main()
