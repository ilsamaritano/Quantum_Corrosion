#!/usr/bin/env python3
"""
run_pipeline.py — Complete Quantum Corrosion Classification Pipeline
=====================================================================
Orchestrates: preprocessing → classical baselines → quantum hybrid
→ data efficiency → ablation → noise robustness → figures.

Usage:
    python -m scripts.run_pipeline                    # full pipeline
    python -m scripts.run_pipeline --stage preprocess  # single stage
    python -m scripts.run_pipeline --stage classical
    python -m scripts.run_pipeline --stage quantum
    python -m scripts.run_pipeline --stage efficiency
    python -m scripts.run_pipeline --stage ablation
    python -m scripts.run_pipeline --stage figures
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from itertools import product

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config, Config
from src.preprocessing import preprocess_dataset
from src.dataset import (
    create_splits, build_dataloaders, PCAExtractor,
    PCASpectrogramDataset, subsample_metadata, reduce_redundancy_metadata, SpectrogramDataset,
    get_val_transform,
)
from src.classical_models import build_classical_model, model_summary, count_parameters
from src.training import train_model, train_multi_round, set_seed
from src.evaluation import (
    evaluate_model, aggregate_rounds, save_metrics_csv,
    save_learning_curves_csv, save_ablation_csv, AggregatedResult, EvalResult,
    compute_metrics,
)
from src.plots import generate_all_figures, PlotConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/pipeline.log", mode="a"),
    ]
)
logger = logging.getLogger("pipeline")


def _compute_balanced_class_weights(metadata: List[Dict], n_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights normalized to mean=1."""
    labels = np.array([int(r["label"]) for r in metadata], dtype=np.int64)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def _predict_probabilities(
    model: torch.nn.Module,
    loader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (labels, probabilities) for a model on a loader."""
    model = model.to(device)
    model.eval()
    all_probs = []
    all_labels = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)


# ════════════════════════════════════════════════════════════════════════════
# Stage 1: Preprocessing
# ════════════════════════════════════════════════════════════════════════════

def stage_preprocess(cfg: Config) -> Tuple[List, List, List]:
    """Load IQ files, compute spectrograms, split dataset."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Preprocessing")
    logger.info("=" * 60)

    meta_path = cfg.paths.data_processed / "metadata.json"
    if meta_path.exists():
        logger.info(f"Loading cached metadata from {meta_path}")
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = preprocess_dataset(cfg, max_workers=4)

    train_meta, val_meta, test_meta = create_splits(
        metadata,
        ratios=cfg.train.split_ratios,
        seed=cfg.train.seed,
        save_dir=cfg.paths.data_splits,
    )

    logger.info(f"Dataset split: train={len(train_meta)}, "
                f"val={len(val_meta)}, test={len(test_meta)}")
    return train_meta, val_meta, test_meta


def load_splits(cfg: Config) -> Tuple[List, List, List]:
    """Load pre-computed splits from disk."""
    splits = {}
    for name in ["train", "val", "test"]:
        path = cfg.paths.data_splits / f"{name}.json"
        with open(path) as f:
            splits[name] = json.load(f)
    return splits["train"], splits["val"], splits["test"]


# ════════════════════════════════════════════════════════════════════════════
# Stage 2: Classical Baselines
# ════════════════════════════════════════════════════════════════════════════

def stage_classical(
    cfg: Config,
    train_meta: List,
    val_meta: List,
    test_meta: List,
) -> Dict[str, Tuple[AggregatedResult, List[EvalResult]]]:
    """Train and evaluate all classical baselines."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Classical Baselines")
    logger.info("=" * 60)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_meta, val_meta, test_meta,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )

    all_aggregated = {}

    for arch in cfg.classical.architectures:
        logger.info(f"\n{'─'*40}\nTraining: {arch}\n{'─'*40}")

        def model_factory():
            return build_classical_model(
                arch, cfg.n_classes, cfg.classical.in_channels,
                cfg.classical.pretrained, cfg.classical.dropout,
            )

        # Print model summary
        sample_model = model_factory()
        logger.info(model_summary(sample_model, arch))
        del sample_model

        # Multi-round training
        round_results = train_multi_round(
            model_factory, train_loader, val_loader,
            n_rounds=cfg.train.n_rounds,
            base_seed=cfg.train.seed,
            epochs=cfg.train.epochs,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            device=cfg.train.device,
            scheduler_name=cfg.train.scheduler,
            patience=cfg.train.patience,
            use_amp=cfg.train.mixed_precision,
            grad_clip=cfg.train.gradient_clip,
            label_smoothing=cfg.train.label_smoothing,
            save_dir=cfg.paths.models,
            model_name=arch,
        )

        # Evaluate each round
        eval_results = []
        for r, rr in enumerate(round_results):
            model = model_factory()
            ckpt = cfg.paths.models / f"{arch}_round{r}.pt"
            if ckpt.exists():
                state = torch.load(ckpt, map_location=cfg.train.device)
                model.load_state_dict(state["state_dict"])
            er = evaluate_model(
                model, test_loader, cfg.class_names,
                device=cfg.train.device, model_name=arch,
                round_id=r, train_time=rr.train_time_total,
            )
            eval_results.append(er)

        agg = aggregate_rounds(eval_results)
        all_aggregated[arch] = (agg, eval_results)
        logger.info(
            f"  {arch}: mean F1={agg.mean_f1:.4f} ± {agg.std_f1:.4f} "
            f"[{agg.min_f1:.4f}, {agg.max_f1:.4f}] var={agg.var_f1:.2e}"
        )

    return all_aggregated


# ════════════════════════════════════════════════════════════════════════════
# Stage 3: Quantum Hybrid Model
# ════════════════════════════════════════════════════════════════════════════

def stage_quantum(
    cfg: Config,
    train_meta: List,
    val_meta: List,
    test_meta: List,
    resume_checkpoint: str = None,
) -> Dict[str, Tuple[AggregatedResult, List[EvalResult]]]:
    """Train and evaluate hybrid quantum-classical models."""
    logger.info("=" * 60)
    logger.info("STAGE 3: Quantum Hybrid Models")
    logger.info("=" * 60)

    try:
        from src.quantum_models import build_quantum_model, quantum_param_count
    except ImportError:
        logger.warning("PennyLane not available — skipping quantum stage")
        return {}

    train_meta_q = reduce_redundancy_metadata(
        train_meta, stride=cfg.quantum.redundancy_stride
    )
    logger.info(
        f"Quantum train reduction: {len(train_meta)} -> {len(train_meta_q)} "
        f"(stride={cfg.quantum.redundancy_stride})"
    )

    # Fit PCA for dimensionality reduction
    pca_path = cfg.paths.models / "pca_reducer.pkl"
    pca = PCAExtractor(n_components=cfg.quantum.pca_dim)
    if pca_path.exists():
        pca.load(pca_path)
    else:
        pca.fit(train_meta_q)
        pca.save(pca_path)

    # Build PCA-based dataloaders
    from torch.utils.data import DataLoader
    train_ds = PCASpectrogramDataset(train_meta_q, pca, use_feature_engineering=True)
    val_ds = PCASpectrogramDataset(val_meta, pca, use_feature_engineering=True)
    test_ds = PCASpectrogramDataset(test_meta, pca, use_feature_engineering=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=cfg.train.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size,
                            shuffle=False, num_workers=cfg.train.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size,
                             shuffle=False, num_workers=cfg.train.num_workers)

    all_aggregated = {}
    class_weights = None
    if cfg.quantum.use_class_weights:
        class_weights = _compute_balanced_class_weights(train_meta_q, cfg.n_classes)
        logger.info(f"Quantum class weights: {class_weights.tolist()}")

    for encoding in [cfg.quantum.encoding]:
        model_name = f"hybrid_vqc_{encoding}"
        logger.info(f"\n{'─'*40}\nTraining: {model_name}\n{'─'*40}")

        best_cfg = {
            "encoding": encoding,
            "n_layers": cfg.quantum.n_layers,
            "entanglement": cfg.quantum.entanglement,
            "lr": cfg.quantum.lr,
        }

        if cfg.quantum.auto_tune:
            trials = []
            for enc, depth, ent, lr in product(
                cfg.quantum.tune_encodings,
                cfg.quantum.tune_layers,
                cfg.quantum.tune_entanglement,
                cfg.quantum.tune_lrs,
            ):
                trials.append({
                    "encoding": enc,
                    "n_layers": int(depth),
                    "entanglement": ent,
                    "lr": float(lr),
                })
            trials = trials[:max(1, int(cfg.quantum.tuning_max_trials))]

            logger.info(f"Quantum auto-tuning enabled: trying {len(trials)} configs")
            best_val = -1.0
            for i, trial in enumerate(trials, start=1):
                logger.info(
                    f"  [Tune {i}/{len(trials)}] "
                    f"enc={trial['encoding']} depth={trial['n_layers']} "
                    f"ent={trial['entanglement']} lr={trial['lr']:.2e}"
                )

                def tuning_factory(cfg_trial=trial):
                    return build_quantum_model(
                        input_dim=cfg.quantum.pca_dim,
                        n_classes=cfg.n_classes,
                        n_qubits=cfg.quantum.n_qubits,
                        n_layers=cfg_trial["n_layers"],
                        encoding=cfg_trial["encoding"],
                        entanglement=cfg_trial["entanglement"],
                        backend=cfg.quantum.backend,
                        diff_method=cfg.quantum.diff_method,
                        head_dims=cfg.quantum.classical_head,
                    )

                tune_results = train_multi_round(
                    tuning_factory,
                    train_loader,
                    val_loader,
                    n_rounds=1,
                    base_seed=cfg.train.seed,
                    epochs=min(cfg.train.epochs, cfg.quantum.tuning_epochs),
                    lr=trial["lr"],
                    weight_decay=cfg.train.weight_decay,
                    device=cfg.train.device,
                    scheduler_name="cosine",
                    patience=min(cfg.train.patience, 12),
                    use_amp=False,
                    grad_clip=cfg.train.gradient_clip,
                    label_smoothing=cfg.train.label_smoothing,
                    class_weights=class_weights,
                    save_dir=None,
                    model_name=f"{model_name}_tune_{i}",
                )

                trial_val = tune_results[0].best_val_acc
                if trial_val > best_val:
                    best_val = trial_val
                    best_cfg = trial

            logger.info(
                "Best tuned config: "
                f"enc={best_cfg['encoding']} depth={best_cfg['n_layers']} "
                f"ent={best_cfg['entanglement']} lr={best_cfg['lr']:.2e} "
                f"(val_acc={best_val:.4f})"
            )

        def model_factory(
            enc=best_cfg["encoding"],
            depth=best_cfg["n_layers"],
            ent=best_cfg["entanglement"],
            apply_resume=True,
        ):
            model = build_quantum_model(
                input_dim=cfg.quantum.pca_dim,
                n_classes=cfg.n_classes,
                n_qubits=cfg.quantum.n_qubits,
                n_layers=depth,
                encoding=enc,
                entanglement=ent,
                backend=cfg.quantum.backend,
                diff_method=cfg.quantum.diff_method,
                head_dims=cfg.quantum.classical_head,
            )
            if resume_checkpoint and apply_resume:
                logger.info(f"Resuming weights from {resume_checkpoint}")
                state = torch.load(resume_checkpoint, map_location=cfg.train.device)
                model.load_state_dict(state.get("state_dict", state))
            return model

        # Quantum models: honor configured rounds/epochs for full training runs
        n_rounds_q = cfg.train.n_rounds
        round_results = train_multi_round(
            model_factory, train_loader, val_loader,
            n_rounds=n_rounds_q,
            base_seed=cfg.train.seed,
            epochs=cfg.train.epochs,
            lr=best_cfg["lr"],
            weight_decay=cfg.train.weight_decay,
            device=cfg.train.device,  # use configured device
            scheduler_name="cosine",
            patience=cfg.train.patience,
            use_amp=False,
            grad_clip=cfg.train.gradient_clip,
            label_smoothing=cfg.train.label_smoothing,
            class_weights=class_weights,
            save_dir=cfg.paths.models,
            model_name=model_name,
        )

        eval_results = []
        for r, rr in enumerate(round_results):
            model = model_factory(apply_resume=False)
            ckpt = cfg.paths.models / f"{model_name}_round{r}.pt"
            if ckpt.exists():
                state = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(state["state_dict"])
            er = evaluate_model(
                model, test_loader, cfg.class_names,
                device="cpu", model_name=model_name,
                round_id=r, train_time=rr.train_time_total,
            )
            eval_results.append(er)

        agg = aggregate_rounds(eval_results)
        all_aggregated[model_name] = (agg, eval_results)

        # Ensemble over top-k checkpoints from best validation rounds
        top_k = max(1, min(cfg.quantum.ensemble_top_k, len(round_results)))
        ranked_rounds = sorted(
            list(enumerate(round_results)),
            key=lambda x: x[1].best_val_acc,
            reverse=True,
        )[:top_k]
        ckpt_round_ids = [rid for rid, _ in ranked_rounds]

        y_true = None
        prob_sum = None
        valid_members = 0
        for rid in ckpt_round_ids:
            ckpt = cfg.paths.models / f"{model_name}_round{rid}.pt"
            if not ckpt.exists():
                continue

            m = model_factory(apply_resume=False)
            state = torch.load(ckpt, map_location=cfg.train.device)
            m.load_state_dict(state["state_dict"])
            labels, probs = _predict_probabilities(m, test_loader, cfg.train.device)

            if y_true is None:
                y_true = labels
                prob_sum = probs
            else:
                prob_sum += probs
            valid_members += 1

        if valid_members > 1:
            y_pred_ens = np.argmax(prob_sum / valid_members, axis=1)
            ens_result = compute_metrics(
                y_true=y_true,
                y_pred=y_pred_ens,
                class_names=cfg.class_names,
                model_name=f"{model_name}_ensemble",
                round_id=0,
            )
            logger.info(
                f"  [{ens_result.model_name}] "
                f"Acc={ens_result.accuracy:.4f} F1={ens_result.f1_score:.4f} "
                f"members={valid_members}"
            )
            ens_agg = aggregate_rounds([ens_result])
            all_aggregated[ens_result.model_name] = (ens_agg, [ens_result])

        # Log quantum parameter breakdown
        sample_m = model_factory(apply_resume=False)
        qpc = quantum_param_count(sample_m)
        logger.info(
            f"  Quantum params: {qpc['quantum']:,} "
            f"({qpc['quantum_fraction']:.1%} of {qpc['total']:,} total)"
        )

    return all_aggregated


# ════════════════════════════════════════════════════════════════════════════
# Stage 3.5: Ensemble (Classical + Quantum)
# ════════════════════════════════════════════════════════════════════════════

def stage_ensemble(
    cfg: Config,
    classical_results: Dict[str, Tuple[Any, List]],
    quantum_results: Dict[str, Tuple[Any, List]],
    test_meta: List,
    test_loader,
) -> Dict[str, Tuple[AggregatedResult, List[EvalResult]]]:
    """Combine classical and quantum model predictions for improved accuracy."""
    logger.info("=" * 60)
    logger.info("STAGE 3.5: Ensemble (Classical + Quantum Hybrid)")
    logger.info("=" * 60)

    from src.ensemble import EnsembleClassifier, evaluate_ensemble

    ensemble_results = {}

    # Extract best classical model (EfficientNet B0)
    classname_best_classical = None
    best_classical_acc = -1.0
    for arch_name, (agg, evals) in classical_results.items():
        if "efficientnet" in arch_name.lower():
            if agg.mean_accuracy > best_classical_acc:
                best_classical_acc = agg.mean_accuracy
                classname_best_classical = arch_name

    if not classname_best_classical:
        logger.warning("No EfficientNet found in classical results; skipping ensemble")
        return {}

    # Extract best quantum model
    quantum_model_name = None
    best_quantum_acc = -1.0
    for qmodel_name, (agg, evals) in quantum_results.items():
        if "ensemble" not in qmodel_name.lower() and agg.mean_accuracy > best_quantum_acc:
            best_quantum_acc = agg.mean_accuracy
            quantum_model_name = qmodel_name

    if not quantum_model_name:
        logger.warning("No quantum model found; skipping ensemble")
        return {}

    logger.info(
        f"Combining: {classname_best_classical} (Acc={best_classical_acc:.4f}) "
        f"+ {quantum_model_name} (Acc={best_quantum_acc:.4f})"
    )

    # Load best checkpoints
    classical_ckpt = cfg.paths.models / f"{classname_best_classical}_round0.pt"
    quantum_ckpt = cfg.paths.models / f"{quantum_model_name}_round0.pt"

    if not classical_ckpt.exists() or not quantum_ckpt.exists():
        logger.warning(f"Checkpoints not found for ensemble; skipping")
        return {}

    classical_state = torch.load(classical_ckpt, map_location="cpu")
    quantum_state = torch.load(quantum_ckpt, map_location="cpu")

    # Rebuild models
    classical_model = build_classical_model(
        classname_best_classical,
        in_channels=cfg.classical.in_channels,
        n_classes=cfg.n_classes,
        dropout=cfg.classical.dropout,
        pretrained=False,
    )
    classical_model.load_state_dict(classical_state.get("state_dict", classical_state))
    classical_model = classical_model.to(cfg.train.device).eval()

    # Load quantum model
    from src.quantum_models import build_quantum_model
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
    quantum_model.load_state_dict(quantum_state.get("state_dict", quantum_state))
    quantum_model = quantum_model.to(cfg.train.device).eval()

    # Create ensemble with multiple fusion strategies
    for fusion_method in ["weighted_avg"]:  # Can add "voting", "stacking" later
        ensemble_name = f"ensemble_{fusion_method}"
        logger.info(f"\n{'─'*40}\nFusion: {fusion_method}\n{'─'*40}")

        ensemble = EnsembleClassifier(
            classical_model=classical_model,
            quantum_model=quantum_model,
            n_classes=cfg.n_classes,
            fusion_method=fusion_method,
            classical_weight=0.65,
            quantum_weight=0.35,
        )
        ensemble = ensemble.to(cfg.train.device)

        # Evaluate on test set
        ens_metrics = evaluate_ensemble(
            ensemble,
            test_loader,
            device=cfg.train.device,
            n_classes=cfg.n_classes,
        )

        ens_eval = EvalResult(
            model_name=ensemble_name,
            accuracy=ens_metrics['accuracy'],
            precision=ens_metrics['precision'],
            recall=ens_metrics['recall'],
            f1_score=ens_metrics['f1'],
            train_time_sec=0.0,
            inference_time_ms_per_sample=0.0,
            round_id=0,
            confusion_matrix=None,
        )

        ens_agg = aggregate_rounds([ens_eval])
        ensemble_results[ensemble_name] = (ens_agg, [ens_eval])

        logger.info(
            f"  {ensemble_name}: "
            f"Acc={ens_metrics['accuracy']:.4f} "
            f"F1={ens_metrics['f1']:.4f}"
        )

        # Save ensemble checkpoint
        ensemble_ckpt_path = cfg.paths.models / f"{ensemble_name}_round0.pt"
        torch.save({
            "state_dict": ensemble.state_dict(),
            "config": {
                "fusion_method": fusion_method,
                "classical_weight": 0.65,
                "quantum_weight": 0.35,
            }
        }, ensemble_ckpt_path)
        logger.info(f"  Saved ensemble to {ensemble_ckpt_path}")

    return ensemble_results


# ════════════════════════════════════════════════════════════════════════════
# Stage 4: Data Efficiency Experiments
# ════════════════════════════════════════════════════════════════════════════

def stage_data_efficiency(
    cfg: Config,
    train_meta: List,
    val_meta: List,
    test_meta: List,
    selected_models: List[str] = ["resnet34", "efficientnet_b0"],
) -> Dict[str, List[Tuple[float, float]]]:
    """Learning curves: train on data fractions, measure F1."""
    logger.info("=" * 60)
    logger.info("STAGE 4: Data Efficiency")
    logger.info("=" * 60)

    results = {}

    for arch in selected_models:
        logger.info(f"\nData efficiency for: {arch}")
        model_results = []

        for frac in cfg.ablation.data_fractions:
            frac_evals = []
            for rep in range(cfg.ablation.n_repeats):
                seed = cfg.train.seed + rep * 100
                set_seed(seed)
                sub_meta = subsample_metadata(train_meta, frac, seed=seed)

                sub_loader, val_loader, test_loader = build_dataloaders(
                    sub_meta, val_meta, test_meta,
                    batch_size=cfg.train.batch_size,
                    num_workers=cfg.train.num_workers,
                )

                model = build_classical_model(
                    arch, cfg.n_classes, cfg.classical.in_channels,
                )
                rr = train_model(
                    model, sub_loader, val_loader,
                    epochs=min(cfg.train.epochs, 80),
                    lr=cfg.train.lr, device=cfg.train.device,
                    use_amp=cfg.train.mixed_precision,
                    patience=15, round_id=rep,
                    model_name=f"{arch}_frac{frac}_rep{rep}",
                )
                er = evaluate_model(
                    model, test_loader, cfg.class_names,
                    device=cfg.train.device, model_name=arch,
                )
                frac_evals.append(er)

            f1s = [e.f1_score for e in frac_evals]
            model_results.append((np.mean(f1s), np.std(f1s)))
            logger.info(
                f"  {arch} @ {frac*100:.0f}%: "
                f"F1={np.mean(f1s):.4f} ± {np.std(f1s):.4f}"
            )

        results[arch] = model_results

    return results


# ════════════════════════════════════════════════════════════════════════════
# Stage 5: Ablation Study
# ════════════════════════════════════════════════════════════════════════════

def stage_ablation(
    cfg: Config,
    train_meta: List,
    val_meta: List,
    test_meta: List,
) -> Dict[str, List[float]]:
    """Ablation: vary encoding, circuit depth, qubit count."""
    logger.info("=" * 60)
    logger.info("STAGE 5: Ablation Study")
    logger.info("=" * 60)

    try:
        from src.quantum_models import build_quantum_model
    except ImportError:
        logger.warning("PennyLane not available — skipping ablation")
        return {}, []

    pca_path = cfg.paths.models / "pca_reducer.pkl"
    pca = PCAExtractor(n_components=cfg.quantum.pca_dim)
    if pca_path.exists():
        pca.load(pca_path)
    else:
        pca.fit(train_meta)
        pca.save(pca_path)

    from torch.utils.data import DataLoader
    train_ds = PCASpectrogramDataset(train_meta, pca, use_feature_engineering=True)
    val_ds = PCASpectrogramDataset(val_meta, pca, use_feature_engineering=True)
    test_ds = PCASpectrogramDataset(test_meta, pca, use_feature_engineering=True)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    ablation_results = []

    # Vary encoding type
    for enc in cfg.ablation.encodings:
        try:
            model = build_quantum_model(
                input_dim=cfg.quantum.pca_dim, n_classes=cfg.n_classes,
                n_qubits=cfg.quantum.n_qubits, n_layers=cfg.quantum.n_layers,
                encoding=enc,
            )
            rr = train_model(
                model, train_loader, val_loader,
                epochs=40, lr=cfg.quantum.lr, device="cpu",
                use_amp=False, patience=15, model_name=f"ablation_{enc}",
            )
            er = evaluate_model(model, test_loader, cfg.class_names, device="cpu")
            ablation_results.append({
                "variable": "encoding", "value": enc,
                "f1": er.f1_score, "accuracy": er.accuracy,
            })
        except Exception as e:
            logger.error(f"Ablation failed for encoding={enc}: {e}")

    # Vary circuit depth
    for depth in cfg.ablation.circuit_depths:
        try:
            model = build_quantum_model(
                input_dim=cfg.quantum.pca_dim, n_classes=cfg.n_classes,
                n_qubits=cfg.quantum.n_qubits, n_layers=depth,
                encoding=cfg.quantum.encoding,
            )
            rr = train_model(
                model, train_loader, val_loader,
                epochs=40, lr=cfg.quantum.lr, device="cpu",
                use_amp=False, patience=15, model_name=f"ablation_depth{depth}",
            )
            er = evaluate_model(model, test_loader, cfg.class_names, device="cpu")
            ablation_results.append({
                "variable": "depth", "value": depth,
                "f1": er.f1_score, "accuracy": er.accuracy,
            })
        except Exception as e:
            logger.error(f"Ablation failed for depth={depth}: {e}")

    save_ablation_csv(ablation_results, cfg.paths.metrics / "ablation_results.csv")
    return ablation_results


# ════════════════════════════════════════════════════════════════════════════
# Stage 6: Generate Figures
# ════════════════════════════════════════════════════════════════════════════

def stage_figures(
    cfg: Config,
    all_aggregated: List[AggregatedResult],
    test_meta: List = None,
    learning_curve_data: Dict = None,
    ablation_data: Dict = None,
    confusion_matrices: Dict = None,
):
    """Generate all publication-quality PDF figures."""
    logger.info("=" * 60)
    logger.info("STAGE 6: Generating Figures")
    logger.info("=" * 60)

    # Collect example spectrograms
    example_specs = {}
    if test_meta:
        for cls_idx, cls_name in enumerate(cfg.class_names[:3]):
            for rec in test_meta:
                if rec["label"] == cls_idx:
                    example_specs[cls_name] = np.load(rec["path"])
                    break

    generate_all_figures(
        aggregated=all_aggregated,
        out_dir=cfg.paths.figures,
        cfg=cfg.plot,
        example_spectrograms=example_specs if example_specs else None,
        learning_curve_data=learning_curve_data,
        learning_curve_fractions=cfg.ablation.data_fractions if learning_curve_data else None,
        confusion_matrices=confusion_matrices,
    )


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Quantum Corrosion Pipeline")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "preprocess", "classical", "quantum",
                                 "ensemble", "efficiency", "ablation", "figures"],
                        help="Pipeline stage to run")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Path to raw IQ data directory")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Path to a saved .pt checkpoint to resume training from")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-auto-tune", action="store_true",
                        help="Disable quantum auto-tuning and use configured quantum params directly")
    parser.add_argument("--force-preprocess", action="store_true",
                        help="Recompute spectrogram metadata even if a cached file exists")
    args = parser.parse_args()

    # Build configuration
    overrides = {}
    if args.data_dir:
        overrides["paths.data_raw"] = Path(args.data_dir)
    if args.device:
        overrides["train.device"] = args.device
    if args.epochs:
        overrides["train.epochs"] = args.epochs
    if args.rounds:
        overrides["train.n_rounds"] = args.rounds
    if args.batch_size:
        overrides["train.batch_size"] = args.batch_size
    if args.no_auto_tune:
        overrides["quantum.auto_tune"] = False

    cfg = get_config(**overrides)

    logger.info("=" * 60)
    logger.info("Quantum Corrosion Classification Pipeline")
    logger.info(f"Device: {cfg.train.device}")
    logger.info(f"Classes: {cfg.class_names}")
    logger.info(f"Data: {cfg.paths.data_raw}")
    logger.info("=" * 60)

    t_start = time.time()
    all_aggregated_results = []
    confusion_matrices = {}
    learning_curve_data = None
    classical_results = {}
    quantum_results = {}

    # ── Preprocessing ─────────────────────────────────────────────────
    if args.stage in ("all", "preprocess"):
        if args.force_preprocess:
            meta_path = cfg.paths.data_processed / "metadata.json"
            if meta_path.exists():
                logger.info(f"Force preprocessing enabled; ignoring cached metadata at {meta_path}")
                meta_path.unlink()
        train_meta, val_meta, test_meta = stage_preprocess(cfg)
    else:
        train_meta, val_meta, test_meta = load_splits(cfg)

    # ── Classical Baselines ───────────────────────────────────────────
    if args.stage in ("all", "classical"):
        classical_results = stage_classical(cfg, train_meta, val_meta, test_meta)
        for arch, (agg, evals) in classical_results.items():
            all_aggregated_results.append(agg)
            # Best round confusion matrix
            best_eval = max(evals, key=lambda e: e.f1_score)
            if best_eval.confusion_matrix is not None:
                confusion_matrices[arch] = (best_eval.confusion_matrix, cfg.class_names)

    # ── Quantum Hybrid ────────────────────────────────────────────────
    if args.stage in ("all", "quantum"):
        quantum_results = stage_quantum(
            cfg, train_meta, val_meta, test_meta,
            resume_checkpoint=args.resume_checkpoint
        )
        for name, (agg, evals) in quantum_results.items():
            all_aggregated_results.append(agg)
            best_eval = max(evals, key=lambda e: e.f1_score)
            if best_eval.confusion_matrix is not None:
                confusion_matrices[name] = (best_eval.confusion_matrix, cfg.class_names)

    # ── Ensemble (Classical + Quantum) ────────────────────────────────
    if args.stage in ("all", "ensemble"):
        # Need results from both classical and quantum
        if not classical_results:
            classical_results = stage_classical(cfg, train_meta, val_meta, test_meta)
            for arch, (agg, evals) in classical_results.items():
                all_aggregated_results.append(agg)
        if not quantum_results:
            quantum_results = stage_quantum(cfg, train_meta, val_meta, test_meta)
            for name, (agg, evals) in quantum_results.items():
                all_aggregated_results.append(agg)

        # Build test loader for ensemble evaluation
        from torch.utils.data import DataLoader
        test_ds = SpectrogramDataset(test_meta, transform=get_val_transform())
        test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size,
                                 shuffle=False, num_workers=cfg.train.num_workers)

        ensemble_results = stage_ensemble(cfg, classical_results, quantum_results, test_meta, test_loader)
        for name, (agg, evals) in ensemble_results.items():
            all_aggregated_results.append(agg)

    # ── Data Efficiency ───────────────────────────────────────────────
    if args.stage in ("all", "efficiency"):
        lc_results = stage_data_efficiency(
            cfg, train_meta, val_meta, test_meta,
            selected_models=["resnet34", "efficientnet_b0"],
        )
        learning_curve_data = lc_results

    # ── Ablation ──────────────────────────────────────────────────────
    if args.stage in ("all", "ablation"):
        ablation_results = stage_ablation(cfg, train_meta, val_meta, test_meta)

    # ── Save aggregated metrics ───────────────────────────────────────
    if all_aggregated_results:
        save_metrics_csv(all_aggregated_results,
                         cfg.paths.metrics / "metrics.csv")

    # ── Generate Figures ──────────────────────────────────────────────
    if args.stage in ("all", "figures"):
        stage_figures(
            cfg, all_aggregated_results, test_meta,
            learning_curve_data=learning_curve_data,
            confusion_matrices=confusion_matrices,
        )

    elapsed = time.time() - t_start
    logger.info(f"\nPipeline complete in {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
