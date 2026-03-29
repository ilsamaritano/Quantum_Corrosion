#!/usr/bin/env python3
"""Train the hybrid quantum-classical classifier on PCA-reduced IQ features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.classical_models import count_trainable_params
from src.data_loading import load_splits
from src.dimensionality_reduction import fit_transform_pca, save_reducer
from src.evaluation import evaluate_model
from src.metrics import compute_classification_metrics, save_metrics_csv
from src.quantum_models import HybridQuantumClassifier, train_quantum_model
from src.training import load_checkpoint
from src.utils import ensure_dir, save_json, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train hybrid quantum-classical classifier."
    )
    parser.add_argument("--data_dir", default="data/splits/tabular",
                        help="Directory with tabular train/val/test splits.")
    parser.add_argument("--config", default="configs/model_quantum.yaml")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--encoding", default=None,
                        help="Override encoding: angle or amplitude.")
    parser.add_argument("--n_qubits", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=None,
                        help="PCA components (latent dimension).")
    parser.add_argument("--device", default="cpu")
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
    qcfg = cfg.get("hybrid_quantum", {})

    # CLI overrides
    n_qubits = args.n_qubits or qcfg.get("n_qubits", 4)
    n_layers = args.n_layers or qcfg.get("n_layers", 3)
    encoding = args.encoding or qcfg.get("encoding", "angle")
    latent_dim = args.latent_dim or qcfg.get("latent_dim", 16)
    n_epochs = qcfg.get("n_epochs", 50)
    lr = qcfg.get("lr", 0.01)
    batch_size = qcfg.get("batch_size", 16)
    patience = qcfg.get("patience", 20)

    print(f"Config: n_qubits={n_qubits}, n_layers={n_layers}, "
          f"encoding={encoding}, latent_dim={latent_dim}")

    # Load tabular splits
    data_dir = Path(args.data_dir)
    if not (data_dir / "X_train.npy").exists():
        # Fall back to spectrograms flattened
        print(f"Tabular splits not found at {data_dir}, using spectrogram splits …")
        data_dir = Path(args.data_dir).parent
        if not (data_dir / "X_train.npy").exists():
            raise FileNotFoundError(
                f"No splits found. Run scripts/prepare_data.py first."
            )

    splits = load_splits(data_dir)
    n_classes = int(splits["y_train"].max()) + 1

    # Flatten if multi-dimensional (e.g. spectrograms)
    X_train = splits["X_train"].reshape(len(splits["X_train"]), -1).astype(np.float32)
    X_val = splits["X_val"].reshape(len(splits["X_val"]), -1).astype(np.float32)
    X_test = splits["X_test"].reshape(len(splits["X_test"]), -1).astype(np.float32)
    y_train = splits["y_train"].astype(np.int64)
    y_val = splits["y_val"].astype(np.int64)
    y_test = splits["y_test"].astype(np.int64)

    print(f"Loaded splits — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # PCA dimensionality reduction
    print(f"Applying PCA to {latent_dim} components …")
    X_train_pca, pca = fit_transform_pca(X_train, n_components=latent_dim)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir / "metrics")
    ensure_dir(output_dir / "models")

    save_reducer(pca, output_dir / "models" / "pca_reducer.pkl")

    # Build data loaders
    def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train_pca, y_train, shuffle=True)
    val_loader = make_loader(X_val_pca, y_val, shuffle=False)
    test_loader = make_loader(X_test_pca, y_test, shuffle=False)

    # Build model
    print("Building HybridQuantumClassifier …")
    model = HybridQuantumClassifier(
        input_dim=latent_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_classes=n_classes,
        encoding=encoding,
    )
    n_params = count_trainable_params(model)
    print(f"Model has {n_params:,} trainable parameters.")

    # Train
    print(f"\nTraining for up to {n_epochs} epochs …")
    history = train_quantum_model(
        model,
        train_loader,
        val_loader,
        n_epochs=n_epochs,
        lr=lr,
        optimizer_name="adam",
        device=args.device,
    )

    # Save model
    model_path = output_dir / "models" / "hybrid_quantum_best.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluate
    res = evaluate_model(model, test_loader, device=args.device, return_probs=False)
    metrics = compute_classification_metrics(res["y_true"], res["y_pred"])
    metrics["n_params"] = n_params
    metrics["model"] = "hybrid_quantum"
    metrics["n_qubits"] = n_qubits
    metrics["n_layers"] = n_layers
    metrics["encoding"] = encoding
    metrics["latent_dim"] = latent_dim

    print(
        f"\n[hybrid_quantum] Test Accuracy: {metrics['accuracy']:.4f} | "
        f"F1: {metrics['f1']:.4f} | Params: {n_params:,}"
    )

    save_json(history, output_dir / "metrics" / "hybrid_quantum_history.json")
    save_json(metrics, output_dir / "metrics" / "hybrid_quantum_metrics.json")

    import pandas as pd

    df = pd.DataFrame([metrics]).set_index("model")
    save_metrics_csv(df, output_dir / "metrics" / "quantum_model_metrics.csv")
    print(f"Metrics saved to {output_dir / 'metrics'}")


if __name__ == "__main__":
    main()
