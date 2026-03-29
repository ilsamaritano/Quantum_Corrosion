#!/usr/bin/env python3
"""Train classical models (ResNet-34, MobileNetV2, SimpleCNN) on processed data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.classical_models import (
    build_mobilenetv2,
    build_resnet34,
    build_simple_cnn,
    count_trainable_params,
)
from src.data_loading import load_splits
from src.evaluation import evaluate_model
from src.metrics import compute_classification_metrics, save_metrics_csv
from src.training import load_checkpoint, train_model
from src.utils import ensure_dir, save_json, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical CNN models.")
    parser.add_argument("--data_dir", default="data/splits",
                        help="Directory with train/val/test .npy splits.")
    parser.add_argument("--config", default="configs/model_classical.yaml")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument(
        "--model",
        default="all",
        choices=["resnet34", "mobilenetv2", "simple_cnn", "all"],
    )
    parser.add_argument("--device", default=None,
                        help="Torch device (default: auto-detect).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def make_loaders(
    splits: dict,
    batch_size: int,
    n_input_channels: int = 1,
    target_h: int = 64,
    target_w: int = 64,
) -> tuple:
    """Build PyTorch DataLoaders from numpy splits, reshaping as needed."""
    from src.spectrograms import resize_spectrogram

    def prep(X: np.ndarray) -> torch.Tensor:
        # X shape: (N, H, W) — add channel dim and optionally resize
        out = []
        for s in X:
            if s.ndim == 2:
                s = resize_spectrogram(s, target_size=(target_h, target_w))
            out.append(s)
        X_r = np.stack(out).astype(np.float32)
        if X_r.ndim == 3:
            X_r = X_r[:, np.newaxis, :, :]  # (N,1,H,W)
        if n_input_channels == 3 and X_r.shape[1] == 1:
            X_r = np.repeat(X_r, 3, axis=1)
        return torch.from_numpy(X_r)

    X_tr = prep(splits["X_train"])
    y_tr = torch.from_numpy(splits["y_train"].astype(np.int64))
    X_va = prep(splits["X_val"])
    y_va = torch.from_numpy(splits["y_val"].astype(np.int64))
    X_te = prep(splits["X_test"])
    y_te = torch.from_numpy(splits["y_test"].astype(np.int64))

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_and_evaluate(
    model_name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: dict,
    output_dir: Path,
    device: str,
) -> dict:
    """Train a single model and return its test metrics."""
    checkpoint_dir = output_dir / "models"
    ensure_dir(checkpoint_dir)

    history = train_model(
        model,
        train_loader,
        val_loader,
        n_epochs=cfg.get("n_epochs", 50),
        lr=cfg.get("lr", 1e-3),
        optimizer=cfg.get("optimizer", "adam"),
        scheduler=cfg.get("scheduler", "cosine"),
        patience=cfg.get("patience", 15),
        checkpoint_dir=checkpoint_dir,
        device=device,
        model_name=model_name,
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    # Reload best checkpoint
    best_ckpt = checkpoint_dir / f"{model_name}_best.pt"
    if best_ckpt.exists():
        model = load_checkpoint(model, best_ckpt, device=device)

    res = evaluate_model(model, test_loader, device=device, return_probs=False)
    metrics = compute_classification_metrics(res["y_true"], res["y_pred"])
    metrics["n_params"] = count_trainable_params(model)
    metrics["model"] = model_name

    # Save history and metrics
    save_json(history, output_dir / "metrics" / f"{model_name}_history.json")
    save_json(metrics, output_dir / "metrics" / f"{model_name}_metrics.json")

    print(
        f"\n[{model_name}] Test Accuracy: {metrics['accuracy']:.4f} | "
        f"F1: {metrics['f1']:.4f} | Params: {metrics['n_params']:,}"
    )
    return metrics


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    cfg: dict = {}
    if Path(args.config).exists():
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)

    # Load splits
    splits = load_splits(args.data_dir)
    n_classes = int(splits["y_train"].max()) + 1
    print(f"Loaded splits — train: {len(splits['X_train'])}, "
          f"val: {len(splits['X_val'])}, test: {len(splits['X_test'])}")
    print(f"Number of classes: {n_classes}")

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir / "metrics")

    models_to_run = (
        ["resnet34", "mobilenetv2", "simple_cnn"]
        if args.model == "all"
        else [args.model]
    )

    all_metrics: dict = {}

    for mname in models_to_run:
        print(f"\n{'='*60}")
        print(f"Training: {mname}")
        print(f"{'='*60}")
        mcfg = cfg.get(mname, {})
        n_ch = mcfg.get("input_channels", 1)
        bs = mcfg.get("batch_size", 32)

        # Build loaders with appropriate channel count
        train_loader, val_loader, test_loader = make_loaders(
            splits, batch_size=bs, n_input_channels=n_ch
        )

        if mname == "resnet34":
            model = build_resnet34(
                num_classes=n_classes,
                pretrained=mcfg.get("pretrained", False),
                input_channels=n_ch,
            )
        elif mname == "mobilenetv2":
            model = build_mobilenetv2(
                num_classes=n_classes,
                pretrained=mcfg.get("pretrained", False),
                input_channels=n_ch,
            )
        else:  # simple_cnn
            model = build_simple_cnn(
                num_classes=n_classes,
                input_channels=n_ch,
            )

        metrics = train_and_evaluate(
            mname, model, train_loader, val_loader, test_loader,
            mcfg, output_dir, device,
        )
        all_metrics[mname] = metrics

    # Save combined comparison CSV
    import pandas as pd

    df = pd.DataFrame(list(all_metrics.values())).set_index("model")
    save_metrics_csv(df, output_dir / "metrics" / "classical_models_comparison.csv")
    print(f"\nAll classical model metrics saved to {output_dir / 'metrics'}")


if __name__ == "__main__":
    main()
