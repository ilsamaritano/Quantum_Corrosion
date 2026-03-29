"""Model evaluation utilities: inference, confusion matrix, timing."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: "nn.Module",
    data_loader: "torch.utils.data.DataLoader",
    device: str = "cpu",
    return_probs: bool = False,
) -> Dict:
    """Evaluate a PyTorch model on a data loader.

    Args:
        model: Trained PyTorch model.
        data_loader: Data loader yielding ``(X, y)`` batches.
        device: Torch device string.
        return_probs: If ``True``, also return softmax probabilities.

    Returns:
        Dictionary with keys ``"y_true"``, ``"y_pred"``, and optionally
        ``"y_proba"``.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for evaluate_model.")

    import torch

    model.eval().to(device)
    all_preds: List[int] = []
    all_true: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            X_batch, y_batch = batch[0].to(device).float(), batch[1]
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy().tolist())
            if return_probs:
                all_probs.append(probs)

    result = {
        "y_true": np.array(all_true),
        "y_pred": np.array(all_preds),
    }
    if return_probs:
        result["y_proba"] = np.vstack(all_probs)
    return result


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
) -> np.ndarray:
    """Compute (and optionally normalise) the confusion matrix.

    Args:
        y_true: True integer labels.
        y_pred: Predicted integer labels.
        class_names: Optional class name list for logging.
        normalize: If ``True``, normalise rows to sum to 1.

    Returns:
        2-D integer (or float) confusion matrix.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    if class_names:
        logger.debug("Confusion matrix classes: %s", class_names)
    return cm


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def evaluate_all_models(
    models_dict: Dict[str, "nn.Module"],
    test_loader: "torch.utils.data.DataLoader",
    device: str = "cpu",
) -> "pandas.DataFrame":
    """Evaluate all models in *models_dict* and return a comparison DataFrame.

    Args:
        models_dict: ``{model_name: model}`` mapping.
        test_loader: Test data loader.
        device: Torch device string.

    Returns:
        :class:`pandas.DataFrame` with one row per model.
    """
    import pandas as pd

    from .metrics import compute_classification_metrics

    rows = []
    for name, model in models_dict.items():
        res = evaluate_model(model, test_loader, device=device, return_probs=False)
        metrics = compute_classification_metrics(res["y_true"], res["y_pred"])
        metrics["model"] = name
        rows.append(metrics)
        logger.info("Evaluated model '%s': accuracy=%.4f", name, metrics.get("accuracy", 0))

    df = pd.DataFrame(rows).set_index("model")
    return df


# ---------------------------------------------------------------------------
# Inference timing
# ---------------------------------------------------------------------------

def time_inference(
    model: "nn.Module",
    data_loader: "torch.utils.data.DataLoader",
    device: str = "cpu",
    n_warmup: int = 5,
) -> Dict:
    """Measure model inference latency.

    A few warm-up batches are run to stabilise timing before measurements.

    Args:
        model: Trained PyTorch model.
        data_loader: Data loader for timing.
        device: Torch device string.
        n_warmup: Number of warm-up batches.

    Returns:
        Dictionary with ``"mean_batch_ms"``, ``"mean_sample_ms"``,
        ``"total_samples"``.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for time_inference.")

    import torch

    model.eval().to(device)
    batch_times: List[float] = []
    batch_sizes: List[int] = []

    with torch.no_grad():
        for i, (X_batch, _) in enumerate(data_loader):
            X_batch = X_batch.to(device).float()
            if i < n_warmup:
                model(X_batch)
                continue
            t0 = time.perf_counter()
            model(X_batch)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            batch_times.append((t1 - t0) * 1000)
            batch_sizes.append(len(X_batch))

    total_samples = sum(batch_sizes)
    if not batch_times:
        return {"mean_batch_ms": 0.0, "mean_sample_ms": 0.0, "total_samples": 0}

    mean_batch_ms = float(np.mean(batch_times))
    mean_sample_ms = float(
        np.sum([bt for bt in batch_times])
        / (total_samples if total_samples else 1)
    )
    return {
        "mean_batch_ms": mean_batch_ms,
        "mean_sample_ms": mean_sample_ms,
        "total_samples": total_samples,
    }


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

def memory_usage(
    model: "nn.Module",
    input_size: tuple,
    device: str = "cpu",
) -> Dict:
    """Estimate peak memory used by a forward pass.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape, e.g. ``(1, 3, 64, 64)``.
        device: ``"cuda"`` for GPU memory tracking; ``"cpu"`` gives parameter
            counts only.

    Returns:
        Dictionary with ``"param_bytes"``, ``"param_mb"``, and
        ``"peak_gpu_mb"`` (0 when not on CUDA).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for memory_usage.")

    import torch

    param_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    result = {
        "param_bytes": param_bytes,
        "param_mb": param_bytes / (1024 ** 2),
        "peak_gpu_mb": 0.0,
    }

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        x = torch.zeros(*input_size).to(device)
        model.to(device)
        with torch.no_grad():
            try:
                model(x)
            except Exception:
                pass
        result["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return result
