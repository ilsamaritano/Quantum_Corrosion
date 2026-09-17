#!/usr/bin/env python3
"""
Search for a pure-quantum configuration that reaches target accuracy.

This script uses only quantum checkpoints/models (no classical ensemble)
and stops early as soon as target test accuracy is reached.
"""

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.dataset import PCAExtractor, PCASpectrogramDataset, subsample_metadata
from src.evaluation import evaluate_model
from src.quantum_models import build_quantum_model
from src.training import train_model, set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("quantum_pure_search")


@dataclass
class Candidate:
    name: str
    n_qubits: int
    n_layers: int
    encoding: str
    entanglement: str
    lr: float
    head_dims: List[int]
    use_feature_engineering: bool


def load_splits_json(cfg) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    with open(cfg.paths.data_splits / "train.json") as f:
        train_meta = json.load(f)
    with open(cfg.paths.data_splits / "val.json") as f:
        val_meta = json.load(f)
    with open(cfg.paths.data_splits / "test.json") as f:
        test_meta = json.load(f)
    return train_meta, val_meta, test_meta


def make_loaders(cfg, train_meta, val_meta, test_meta, pca, use_feature_engineering: bool):
    train_ds = PCASpectrogramDataset(train_meta, pca, use_feature_engineering=use_feature_engineering)
    val_ds = PCASpectrogramDataset(val_meta, pca, use_feature_engineering=use_feature_engineering)
    test_ds = PCASpectrogramDataset(test_meta, pca, use_feature_engineering=use_feature_engineering)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    return train_loader, val_loader, test_loader


def default_candidates() -> List[Candidate]:
    return [
        Candidate("amp_q8_l10_full_fe", 8, 10, "amplitude", "full", 3e-3, [512, 256, 128], True),
        Candidate("amp_q8_l10_full_raw", 8, 10, "amplitude", "full", 3e-3, [512, 256, 128], False),
        Candidate("amp_q10_l8_full_fe", 10, 8, "amplitude", "full", 1e-3, [512, 256], True),
        Candidate("amp_q10_l10_full_fe", 10, 10, "amplitude", "full", 1e-3, [512, 256], True),
        Candidate("amp_q10_l8_linear_fe", 10, 8, "amplitude", "linear", 1e-3, [512, 256], True),
        Candidate("angle_q8_l12_full_fe", 8, 12, "angle", "full", 1e-3, [512, 256], True),
        Candidate("angle_q10_l10_full_fe", 10, 10, "angle", "full", 1e-3, [512, 256], True),
        Candidate("iqp_q10_l10_full_fe", 10, 10, "iqp", "full", 1e-3, [512, 256], True),
    ]


def main():
    parser = argparse.ArgumentParser(description="Pure-quantum over-90 search")
    parser.add_argument("--target-acc", type=float, default=0.90)
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=str, default="results/quantum_pure_search_summary.csv")
    args = parser.parse_args()

    cfg = get_config(
        **{
            "train.device": args.device,
            "train.batch_size": args.batch_size,
        }
    )

    set_seed(args.seed)
    train_meta, val_meta, test_meta = load_splits_json(cfg)
    if args.train_fraction < 1.0:
        train_meta = subsample_metadata(train_meta, args.train_fraction, seed=args.seed)
        logger.info(
            f"Using train fraction {args.train_fraction:.3f}: {len(train_meta)} training samples"
        )

    pca_suffix = f"frac{str(args.train_fraction).replace('.', 'p')}"
    pca_path = cfg.paths.models / f"pca_reducer_{pca_suffix}.pkl"
    pca_dim = min(128, max(32, len(train_meta) - 1))
    if not pca_path.exists():
        logger.info(f"Fitting PCA on current train subset and saving to {pca_path}")
        pca = PCAExtractor(n_components=pca_dim)
        pca.fit(train_meta)
        pca.save(pca_path)
    else:
        pca = PCAExtractor(n_components=1024).load(pca_path)
        pca_dim = int(getattr(pca.pca, "n_components_", pca.n_components))

    candidates = default_candidates()
    deadline = time.time() + args.max_hours * 3600.0

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    reached = False

    for i, cand in enumerate(candidates, start=1):
        if time.time() >= deadline:
            logger.warning("Time budget exhausted before target was reached.")
            break

        logger.info("=" * 80)
        logger.info(
            f"[{i}/{len(candidates)}] {cand.name} | enc={cand.encoding} q={cand.n_qubits} "
            f"layers={cand.n_layers} ent={cand.entanglement} lr={cand.lr:.2e} "
            f"feat_eng={cand.use_feature_engineering}"
        )

        train_loader, val_loader, test_loader = make_loaders(
            cfg, train_meta, val_meta, test_meta, pca, cand.use_feature_engineering
        )

        model = build_quantum_model(
            input_dim=pca_dim,
            n_classes=cfg.n_classes,
            n_qubits=cand.n_qubits,
            n_layers=cand.n_layers,
            encoding=cand.encoding,
            entanglement=cand.entanglement,
            backend="default.qubit",
            diff_method="backprop",
            head_dims=cand.head_dims,
        )

        save_name = f"quantum_pure_{cand.name}"
        rr = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=cand.lr,
            weight_decay=cfg.train.weight_decay,
            device=args.device,
            scheduler_name="cosine",
            patience=args.patience,
            use_amp=False,
            grad_clip=cfg.train.gradient_clip,
            label_smoothing=0.05,
            class_weights=None,
            round_id=0,
            save_dir=cfg.paths.models,
            model_name=save_name,
        )

        result = evaluate_model(
            model=model,
            test_loader=test_loader,
            class_names=cfg.class_names,
            device=args.device,
            model_name=save_name,
            round_id=0,
            train_time=rr.train_time_total,
        )

        row = {
            "candidate": cand.name,
            "train_fraction": args.train_fraction,
            "encoding": cand.encoding,
            "n_qubits": cand.n_qubits,
            "n_layers": cand.n_layers,
            "entanglement": cand.entanglement,
            "lr": cand.lr,
            "feature_engineering": cand.use_feature_engineering,
            "test_accuracy": result.accuracy,
            "test_f1": result.f1_score,
            "best_val_acc": rr.best_val_acc,
            "best_epoch": rr.best_epoch,
            "train_time_s": rr.train_time_total,
            "checkpoint": str(cfg.paths.models / f"{save_name}_round0.pt"),
        }
        rows.append(row)

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        logger.info(
            f"Result {cand.name}: test_acc={result.accuracy:.4f}, test_f1={result.f1_score:.4f}, "
            f"best_val={rr.best_val_acc:.4f}"
        )

        if result.accuracy >= args.target_acc:
            logger.info("Target reached. Stopping search.")
            reached = True
            break

    if not rows:
        logger.error("No candidate was executed.")
        return 2

    best_row = max(rows, key=lambda x: x["test_accuracy"])
    logger.info("=" * 80)
    logger.info(
        f"Best so far: {best_row['candidate']} | test_acc={best_row['test_accuracy']:.4f} "
        f"test_f1={best_row['test_f1']:.4f}"
    )

    if reached:
        return 0

    logger.warning("Target not reached within configured budget/candidates.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
