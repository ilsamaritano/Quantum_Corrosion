#!/usr/bin/env python3
"""
Few-shot stratified VQC optimization run.

Goal:
- Keep a true VQC model in the loop
- Optimize preprocessing-side representation choices (PCA + feature engineering)
- Search hyperparameters aggressively in few-shot regime
- Stop early if target validation accuracy is reached
"""

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.dataset import PCAExtractor, PCASpectrogramDataset, create_splits
from src.preprocessing import preprocess_dataset
from src.quantum_models import build_quantum_model
from src.training import train_model, set_seed
from src.evaluation import evaluate_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("fewshot_vqc_99")


def load_splits(cfg) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    with open(cfg.paths.data_splits / "train.json") as f:
        train_meta = json.load(f)
    with open(cfg.paths.data_splits / "val.json") as f:
        val_meta = json.load(f)
    with open(cfg.paths.data_splits / "test.json") as f:
        test_meta = json.load(f)
    return train_meta, val_meta, test_meta


def stratified_few_shot(metadata: List[Dict], shots_per_class: int, seed: int) -> List[Dict]:
    rng = np.random.default_rng(seed)
    labels = np.array([int(r["label"]) for r in metadata], dtype=np.int64)
    selected_idx: List[int] = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) == 0:
            continue
        n_keep = min(len(cls_idx), shots_per_class)
        chosen = rng.choice(cls_idx, size=n_keep, replace=False)
        selected_idx.extend(chosen.tolist())

    selected_idx = sorted(selected_idx)
    meta_arr = np.array(metadata)
    return meta_arr[selected_idx].tolist()


def balanced_class_weights(metadata: List[Dict], n_classes: int, device: str) -> torch.Tensor:
    labels = np.array([int(r["label"]) for r in metadata], dtype=np.int64)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def trial_space() -> List[Dict]:
    return [
        {
            "n_qubits": nq,
            "encoding": enc,
            "n_layers": nl,
            "entanglement": ent,
            "lr": lr,
            "pca_dim": pca_dim,
            "feature_engineering": fe,
            "head_dims": head,
            "batch_size": bs,
        }
        for nq in [4, 6, 8]
        for enc in ["angle", "amplitude"]
        for nl in [2, 4, 6]
        for ent in ["linear", "full"]
        for lr in [5e-4, 1e-3, 2e-3]
        for pca_dim in [32, 64]
        for fe in [True, False]
        for head in [[512, 256, 128], [1024, 512, 256]]
        for bs in [16, 32]
    ]


def main():
    parser = argparse.ArgumentParser(description="Few-shot stratified VQC search for high accuracy")
    parser.add_argument("--shots-per-class", type=int, default=64)
    parser.add_argument("--target-accuracy", type=float, default=0.99)
    parser.add_argument("--val-shots-per-class", type=int, default=80)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quantum-backend", default="lightning.gpu")
    parser.add_argument("--diff-method", default="parameter-shift")
    parser.add_argument("--out-dir", default="results/fewshot_vqc_99")
    parser.add_argument("--refresh-preprocess", action="store_true",
                        help="Regenerate processed spectrograms and splits before few-shot search")
    parser.add_argument("--force-preprocess", action="store_true",
                        help="When refreshing, remove existing processed spectrogram files first")
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = get_config()
    cfg.train.device = args.device
    cfg.quantum.backend = args.quantum_backend
    cfg.quantum.diff_method = args.diff_method
    tune_num_workers = 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_preprocess:
        logger.info("Refreshing deterministic preprocessing and data splits...")
        cfg.augment.enabled = False
        if args.force_preprocess:
            for p in cfg.paths.data_processed.glob("*.npy"):
                p.unlink(missing_ok=True)
            meta_path = cfg.paths.data_processed / "metadata.json"
            meta_path.unlink(missing_ok=True)

        metadata = preprocess_dataset(cfg, max_workers=4)
        create_splits(
            metadata,
            ratios=cfg.train.split_ratios,
            seed=cfg.train.seed,
            save_dir=cfg.paths.data_splits,
        )
        logger.info("Preprocessing refresh complete.")

    train_meta, val_meta, test_meta = load_splits(cfg)
    fs_train_meta = stratified_few_shot(train_meta, args.shots_per_class, args.seed)
    fs_val_meta = stratified_few_shot(val_meta, args.val_shots_per_class, args.seed + 1)

    logger.info("=" * 72)
    logger.info("Few-shot stratified VQC optimization")
    logger.info(f"Device: {args.device}")
    logger.info(f"Quantum backend: {cfg.quantum.backend} | diff: {cfg.quantum.diff_method}")
    logger.info(f"Train full: {len(train_meta)} | few-shot: {len(fs_train_meta)}")
    logger.info(
        f"Val full: {len(val_meta)} | Val tune: {len(fs_val_meta)} | Test full: {len(test_meta)}"
    )
    logger.info(f"Target val accuracy: {args.target_accuracy:.4f}")
    logger.info("=" * 72)

    all_trials = trial_space()
    rng = np.random.default_rng(args.seed)
    chosen_idx = rng.choice(len(all_trials), size=min(args.n_trials, len(all_trials)), replace=False)
    chosen_trials = [all_trials[int(i)] for i in chosen_idx]
    pca_cache: Dict[int, Tuple[PCAExtractor, Path]] = {}

    trial_rows: List[Dict] = []
    best = {
        "val_acc": -1.0,
        "val_f1": -1.0,
        "cfg": None,
        "ckpt": None,
        "pca_path": None,
        "trial_id": -1,
    }

    for t_id, tcfg in enumerate(chosen_trials, start=1):
        logger.info("\n" + "-" * 72)
        logger.info(
            f"Trial {t_id}/{len(chosen_trials)} | q={tcfg['n_qubits']} enc={tcfg['encoding']} layers={tcfg['n_layers']} "
            f"ent={tcfg['entanglement']} lr={tcfg['lr']:.1e} pca={tcfg['pca_dim']} "
            f"fe={tcfg['feature_engineering']} bs={tcfg['batch_size']}"
        )

        trial_prefix = f"fewshot99_t{t_id}"
        max_pca = max(8, len(fs_train_meta) - 1)
        pca_dim_eff = min(int(tcfg["pca_dim"]), max_pca)
        if pca_dim_eff != int(tcfg["pca_dim"]):
            logger.info(
                f"Trial {t_id}: reducing PCA dim {tcfg['pca_dim']} -> {pca_dim_eff} "
                f"(few-shot samples={len(fs_train_meta)})"
            )

        if pca_dim_eff in pca_cache:
            pca, pca_path = pca_cache[pca_dim_eff]
        else:
            pca = PCAExtractor(n_components=pca_dim_eff)
            pca.fit(fs_train_meta)
            pca_path = out_dir / f"pca_{pca_dim_eff}.pkl"
            pca.save(pca_path)
            pca_cache[pca_dim_eff] = (pca, pca_path)

        train_ds = PCASpectrogramDataset(
            fs_train_meta,
            pca,
            use_feature_engineering=tcfg["feature_engineering"],
        )
        val_ds = PCASpectrogramDataset(
            fs_val_meta,
            pca,
            use_feature_engineering=tcfg["feature_engineering"],
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=tcfg["batch_size"],
            shuffle=True,
            num_workers=tune_num_workers,
            pin_memory=cfg.train.pin_memory,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=tcfg["batch_size"],
            shuffle=False,
            num_workers=tune_num_workers,
            pin_memory=cfg.train.pin_memory,
        )

        model = build_quantum_model(
            input_dim=pca_dim_eff,
            n_classes=cfg.n_classes,
            n_qubits=tcfg["n_qubits"],
            n_layers=tcfg["n_layers"],
            encoding=tcfg["encoding"],
            entanglement=tcfg["entanglement"],
            backend=cfg.quantum.backend,
            diff_method=cfg.quantum.diff_method,
            head_dims=tcfg["head_dims"],
        )

        class_w = balanced_class_weights(fs_train_meta, cfg.n_classes, args.device)
        t0 = time.time()
        rr = train_model(
            model,
            train_loader,
            val_loader,
            epochs=args.epochs,
            lr=tcfg["lr"],
            weight_decay=cfg.train.weight_decay,
            device=args.device,
            scheduler_name="cosine",
            patience=args.patience,
            use_amp=False,
            grad_clip=cfg.train.gradient_clip,
            label_smoothing=cfg.train.label_smoothing,
            class_weights=class_w,
            round_id=0,
            save_dir=out_dir,
            model_name=trial_prefix,
        )
        elapsed = time.time() - t0

        val_eval = evaluate_model(
            model,
            val_loader,
            cfg.class_names,
            device=args.device,
            model_name=f"{trial_prefix}_val",
            round_id=0,
            train_time=elapsed,
        )
        ckpt = out_dir / f"{trial_prefix}_round0.pt"

        row = {
            "trial_id": t_id,
            **tcfg,
            "pca_dim_effective": pca_dim_eff,
            "shots_per_class": args.shots_per_class,
            "best_epoch": rr.best_epoch,
            "best_val_acc_trainloop": rr.best_val_acc,
            "val_accuracy": val_eval.accuracy,
            "val_f1": val_eval.f1_score,
            "train_time_s": elapsed,
            "checkpoint": str(ckpt),
            "pca_path": str(pca_path),
        }
        trial_rows.append(row)

        improved = (val_eval.accuracy > best["val_acc"]) or (
            abs(val_eval.accuracy - best["val_acc"]) < 1e-12 and val_eval.f1_score > best["val_f1"]
        )
        if improved:
            best.update(
                {
                    "val_acc": val_eval.accuracy,
                    "val_f1": val_eval.f1_score,
                    "cfg": {**dict(tcfg), "pca_dim_effective": pca_dim_eff},
                    "ckpt": str(ckpt),
                    "pca_path": str(pca_path),
                    "trial_id": t_id,
                }
            )

        if val_eval.accuracy >= args.target_accuracy:
            logger.info(f"Target reached on validation at trial {t_id}: {val_eval.accuracy:.4f}")
            break

    csv_path = out_dir / "trials_summary.csv"
    if trial_rows:
        fieldnames = [
            "trial_id",
            "n_qubits",
            "encoding",
            "n_layers",
            "entanglement",
            "lr",
            "pca_dim",
            "pca_dim_effective",
            "feature_engineering",
            "head_dims",
            "batch_size",
            "shots_per_class",
            "best_epoch",
            "best_val_acc_trainloop",
            "val_accuracy",
            "val_f1",
            "train_time_s",
            "checkpoint",
            "pca_path",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trial_rows)

    final_eval = None
    if best["cfg"] is not None and best["ckpt"] is not None and best["pca_path"] is not None:
        best_cfg = best["cfg"]
        pca_best = PCAExtractor(n_components=int(best_cfg["pca_dim_effective"])).load(Path(best["pca_path"]))

        val_full_ds = PCASpectrogramDataset(
            val_meta,
            pca_best,
            use_feature_engineering=best_cfg["feature_engineering"],
        )
        test_full_ds = PCASpectrogramDataset(
            test_meta,
            pca_best,
            use_feature_engineering=best_cfg["feature_engineering"],
        )
        val_full_loader = DataLoader(
            val_full_ds,
            batch_size=best_cfg["batch_size"],
            shuffle=False,
            num_workers=tune_num_workers,
            pin_memory=cfg.train.pin_memory,
        )
        test_full_loader = DataLoader(
            test_full_ds,
            batch_size=best_cfg["batch_size"],
            shuffle=False,
            num_workers=tune_num_workers,
            pin_memory=cfg.train.pin_memory,
        )

        best_model = build_quantum_model(
            input_dim=int(best_cfg["pca_dim_effective"]),
            n_classes=cfg.n_classes,
            n_qubits=int(best_cfg["n_qubits"]),
            n_layers=int(best_cfg["n_layers"]),
            encoding=best_cfg["encoding"],
            entanglement=best_cfg["entanglement"],
            backend=cfg.quantum.backend,
            diff_method=cfg.quantum.diff_method,
            head_dims=best_cfg["head_dims"],
        )
        state = torch.load(best["ckpt"], map_location=args.device)
        best_model.load_state_dict(state.get("state_dict", state))

        val_full_eval = evaluate_model(
            best_model,
            val_full_loader,
            cfg.class_names,
            device=args.device,
            model_name="best_full_val",
            round_id=0,
        )
        test_full_eval = evaluate_model(
            best_model,
            test_full_loader,
            cfg.class_names,
            device=args.device,
            model_name="best_full_test",
            round_id=0,
        )

        final_eval = {
            "val_full_accuracy": val_full_eval.accuracy,
            "val_full_f1": val_full_eval.f1_score,
            "test_full_accuracy": test_full_eval.accuracy,
            "test_full_f1": test_full_eval.f1_score,
        }

    best_path = out_dir / "best_config.json"
    with open(best_path, "w") as f:
        json.dump(
            {
                "target_accuracy": args.target_accuracy,
                "best_val_accuracy": best["val_acc"],
                "best_val_f1": best["val_f1"],
                "best_trial_id": best["trial_id"],
                "best_config": best["cfg"],
                "best_checkpoint": best["ckpt"],
                "best_pca": best["pca_path"],
                "n_trials_executed": len(trial_rows),
                "shots_per_class": args.shots_per_class,
                "val_shots_per_class": args.val_shots_per_class,
                "final_full_eval": final_eval,
            },
            f,
            indent=2,
        )

    logger.info("\n" + "=" * 72)
    logger.info(f"Trials summary saved to {csv_path}")
    logger.info(f"Best config saved to {best_path}")
    logger.info(
        f"Best validation: acc={best['val_acc']:.4f} f1={best['val_f1']:.4f} "
        f"(target={args.target_accuracy:.4f})"
    )
    if best["val_acc"] < args.target_accuracy:
        logger.info("Target not reached in this run; increase n-trials/epochs or shots-per-class.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
