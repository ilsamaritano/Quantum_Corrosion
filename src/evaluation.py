"""
Evaluation — Comprehensive metrics computation
===============================================
Precision, Recall, F1, confusion matrix, inference timing,
parameter counting, and multi-round aggregation.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score,
)
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import csv
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Complete evaluation result for one model on one test set."""
    model_name: str = ""
    round_id: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    n_params_total: int = 0
    n_params_trainable: int = 0
    inference_time_ms: float = 0.0  # per-sample average
    train_time_s: float = 0.0


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Collect all predictions and labels.
    Returns (y_true, y_pred, avg_inference_ms_per_sample).
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_time = 0.0
    total_samples = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        t0 = time.perf_counter()
        outputs = model(inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_time += (t1 - t0)
        total_samples += inputs.size(0)

        _, predicted = outputs.max(1)
        all_preds.append(predicted.cpu().numpy())
        all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    avg_ms = (total_time / total_samples) * 1000.0 if total_samples > 0 else 0.0
    return y_true, y_pred, avg_ms


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str = "",
    round_id: int = 0,
    inference_ms: float = 0.0,
    model: Optional[nn.Module] = None,
    train_time: float = 0.0,
) -> EvalResult:
    """Compute all evaluation metrics."""
    result = EvalResult(
        model_name=model_name,
        round_id=round_id,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1_score=f1_score(y_true, y_pred, average="macro", zero_division=0),
        confusion_matrix=confusion_matrix(y_true, y_pred),
        inference_time_ms=inference_ms,
        train_time_s=train_time,
    )

    # Per-class F1
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        if i < len(per_class):
            result.per_class_f1[name] = float(per_class[i])

    # Parameter count
    if model is not None:
        result.n_params_total = sum(p.numel() for p in model.parameters())
        result.n_params_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

    return result


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: str = "cuda",
    model_name: str = "",
    round_id: int = 0,
    train_time: float = 0.0,
) -> EvalResult:
    """End-to-end evaluation pipeline."""
    model = model.to(device)
    y_true, y_pred, inference_ms = predict(model, test_loader, device)
    result = compute_metrics(
        y_true, y_pred, class_names,
        model_name=model_name, round_id=round_id,
        inference_ms=inference_ms, model=model,
        train_time=train_time,
    )

    # Log results
    logger.info(
        f"  [{model_name} R{round_id}] "
        f"Acc={result.accuracy:.4f} P={result.precision:.4f} "
        f"R={result.recall:.4f} F1={result.f1_score:.4f} "
        f"Inf={result.inference_time_ms:.2f}ms/sample"
    )
    return result


# ── Multi-round aggregation ─────────────────────────────────────────────────

@dataclass
class AggregatedResult:
    model_name: str = ""
    n_rounds: int = 0
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    mean_f1: float = 0.0
    std_f1: float = 0.0
    min_f1: float = 0.0
    max_f1: float = 0.0
    var_f1: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_inference_ms: float = 0.0
    mean_train_time: float = 0.0
    n_params_trainable: int = 0
    per_round_f1: List[float] = field(default_factory=list)


def aggregate_rounds(results: List[EvalResult]) -> AggregatedResult:
    """Aggregate metrics across multiple training rounds."""
    if not results:
        return AggregatedResult()

    f1s = [r.f1_score for r in results]
    agg = AggregatedResult(
        model_name=results[0].model_name,
        n_rounds=len(results),
        mean_accuracy=np.mean([r.accuracy for r in results]),
        std_accuracy=np.std([r.accuracy for r in results]),
        mean_f1=np.mean(f1s),
        std_f1=np.std(f1s),
        min_f1=np.min(f1s),
        max_f1=np.max(f1s),
        var_f1=np.var(f1s),
        mean_precision=np.mean([r.precision for r in results]),
        mean_recall=np.mean([r.recall for r in results]),
        mean_inference_ms=np.mean([r.inference_time_ms for r in results]),
        mean_train_time=np.mean([r.train_time_s for r in results]),
        n_params_trainable=results[0].n_params_trainable,
        per_round_f1=f1s,
    )
    return agg


# ── CSV export ──────────────────────────────────────────────────────────────

def save_metrics_csv(
    aggregated: List[AggregatedResult],
    path: Path,
):
    """Save aggregated results to CSV."""
    fieldnames = [
        "model_name", "n_rounds", "mean_accuracy", "std_accuracy",
        "mean_f1", "std_f1", "min_f1", "max_f1", "var_f1",
        "mean_precision", "mean_recall", "mean_inference_ms",
        "mean_train_time", "n_params_trainable",
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for agg in aggregated:
            row = {k: getattr(agg, k) for k in fieldnames}
            writer.writerow(row)
    logger.info(f"Metrics saved → {path}")


def save_learning_curves_csv(
    results: Dict[str, List[EvalResult]],
    fractions: List[float],
    path: Path,
):
    """Save data-efficiency learning curve results."""
    rows = []
    for model_name, frac_results in results.items():
        for frac, evals in zip(fractions, frac_results):
            f1s = [e.f1_score for e in evals]
            rows.append({
                "model": model_name,
                "fraction": frac,
                "mean_f1": np.mean(f1s),
                "std_f1": np.std(f1s),
                "mean_acc": np.mean([e.accuracy for e in evals]),
                "std_acc": np.std([e.accuracy for e in evals]),
            })
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Learning curves saved → {path}")


def save_ablation_csv(
    results: List[Dict],
    path: Path,
):
    """Save ablation study results."""
    if not results:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Ablation results saved → {path}")
