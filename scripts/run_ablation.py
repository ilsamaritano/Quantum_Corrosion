#!/usr/bin/env python3
"""Ablation study: sweep n_qubits, n_layers and encoding type."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.classical_models import count_trainable_params
from src.data_loading import load_splits
from src.dimensionality_reduction import fit_transform_pca
from src.evaluation import evaluate_model
from src.metrics import compute_classification_metrics, save_metrics_csv
from src.quantum_models import HybridQuantumClassifier, train_quantum_model
from src.utils import ensure_dir, save_json, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quantum ablation study.")
    parser.add_argument("--data_dir", default="data/splits/tabular")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--n_epochs", type=int, default=20,
                        help="Epochs per ablation run (use small value for speed).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    # Load config
    cfg: dict = {}
    if Path(args.config).exists():
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)
    ablation_cfg = cfg.get("experiments", {}).get("ablation", {})

    n_qubits_list = ablation_cfg.get("n_qubits", [2, 4])
    n_layers_list = ablation_cfg.get("n_layers", [1, 2, 3])
    encodings = ablation_cfg.get("encodings", ["angle"])

    # Load data
    data_dir = Path(args.data_dir)
    if not (data_dir / "X_train.npy").exists():
        data_dir = data_dir.parent
    splits = load_splits(data_dir)
    n_classes = int(splits["y_train"].max()) + 1

    X_train = splits["X_train"].reshape(len(splits["X_train"]), -1).astype(np.float32)
    X_val = splits["X_val"].reshape(len(splits["X_val"]), -1).astype(np.float32)
    X_test = splits["X_test"].reshape(len(splits["X_test"]), -1).astype(np.float32)
    y_train = splits["y_train"].astype(np.int64)
    y_val = splits["y_val"].astype(np.int64)
    y_test = splits["y_test"].astype(np.int64)

    latent_dim = min(args.latent_dim, X_train.shape[1])
    X_train_pca, pca = fit_transform_pca(X_train, n_components=latent_dim)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir / "metrics")

    def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y),
        )
        return DataLoader(ds, batch_size=16, shuffle=shuffle)

    train_loader = make_loader(X_train_pca, y_train, True)
    val_loader = make_loader(X_val_pca, y_val, False)
    test_loader = make_loader(X_test_pca, y_test, False)

    rows = []
    combos = list(itertools.product(n_qubits_list, n_layers_list, encodings))
    print(f"Running {len(combos)} ablation combinations …")

    for n_q, n_l, enc in combos:
        print(f"\n--- n_qubits={n_q}, n_layers={n_l}, encoding={enc} ---")
        try:
            model = HybridQuantumClassifier(
                input_dim=latent_dim,
                n_qubits=n_q,
                n_layers=n_l,
                n_classes=n_classes,
                encoding=enc,
            )
            train_quantum_model(
                model, train_loader, val_loader,
                n_epochs=args.n_epochs, lr=0.01, device="cpu",
            )
            res = evaluate_model(model, test_loader, device="cpu")
            metrics = compute_classification_metrics(res["y_true"], res["y_pred"])
            row = {
                "n_qubits": n_q,
                "n_layers": n_l,
                "encoding": enc,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "n_params": count_trainable_params(model),
            }
        except Exception as exc:
            print(f"  FAILED: {exc}")
            row = {
                "n_qubits": n_q, "n_layers": n_l, "encoding": enc,
                "accuracy": float("nan"), "f1": float("nan"), "n_params": 0,
            }
        rows.append(row)
        print(f"  accuracy={row['accuracy']:.4f}, f1={row['f1']:.4f}")

    df = pd.DataFrame(rows)
    out_path = output_dir / "metrics" / "ablation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nAblation results saved to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
