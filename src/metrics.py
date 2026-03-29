"""Classification metrics: accuracy, F1, per-class breakdown, model comparison."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = "macro",
) -> Dict:
    """Compute a comprehensive suite of classification metrics.

    Args:
        y_true: True integer labels, shape ``(N,)``.
        y_pred: Predicted integer labels, shape ``(N,)``.
        y_proba: Optional softmax probability matrix ``(N, n_classes)`` used
            for ROC-AUC computation.
        average: Averaging strategy for multi-class metrics.

    Returns:
        Dictionary with keys: ``accuracy``, ``precision``, ``recall``,
        ``f1``, ``balanced_accuracy``, and optionally ``roc_auc``.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: Dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    if y_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score

            n_classes = y_proba.shape[1]
            if n_classes == 2:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba[:, 1])
                )
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average=average
                    )
                )
        except Exception as exc:
            logger.debug("ROC-AUC computation failed: %s", exc)

    return metrics


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------

def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> "pandas.DataFrame":
    """Compute per-class precision, recall, F1 and support.

    Args:
        y_true: True integer labels.
        y_pred: Predicted integer labels.
        class_names: Optional list of class name strings.

    Returns:
        :class:`pandas.DataFrame` with one row per class.
    """
    import pandas as pd
    from sklearn.metrics import classification_report

    target_names = class_names or [str(i) for i in sorted(np.unique(y_true))]
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    rows = {
        k: v for k, v in report.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    }
    df = pd.DataFrame(rows).T
    df.index.name = "class"
    return df


# ---------------------------------------------------------------------------
# Multi-model summary
# ---------------------------------------------------------------------------

def summarize_model_comparison(
    results_dict: Dict[str, Dict],
    include_params: bool = True,
) -> "pandas.DataFrame":
    """Produce a comparison table for multiple models.

    Args:
        results_dict: ``{model_name: metrics_dict}`` where each metrics dict
            may optionally contain a ``"n_params"`` key.
        include_params: Include parameter count column when available.

    Returns:
        :class:`pandas.DataFrame` with one row per model.
    """
    import pandas as pd

    rows = []
    for model_name, metrics in results_dict.items():
        row = dict(metrics)
        row["model"] = model_name
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    preferred_cols = ["accuracy", "balanced_accuracy", "f1", "precision", "recall"]
    if include_params and "n_params" in df.columns:
        preferred_cols.append("n_params")
    present_cols = [c for c in preferred_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in preferred_cols]
    return df[present_cols + other_cols]


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

def save_metrics_csv(
    metrics_df: "pandas.DataFrame",
    output_path: str | Path,
) -> None:
    """Save a metrics DataFrame to CSV.

    Args:
        metrics_df: DataFrame to save.
        output_path: Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path)
    logger.info("Metrics saved to %s", output_path)


def load_metrics_csv(path: str | Path) -> "pandas.DataFrame":
    """Load a metrics CSV produced by :func:`save_metrics_csv`.

    Args:
        path: Source file path.

    Returns:
        :class:`pandas.DataFrame`.
    """
    import pandas as pd

    df = pd.read_csv(path, index_col=0)
    logger.debug("Loaded metrics from %s: shape %s", path, df.shape)
    return df
