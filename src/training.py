"""Training utilities: epoch loops, early stopping, checkpointing, learning curves."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Monitor a validation metric and signal when training should stop.

    Args:
        patience: How many epochs to wait for improvement before stopping.
        min_delta: Minimum absolute improvement to count as improvement.
        mode: ``"min"`` for loss-like metrics, ``"max"`` for accuracy-like.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, metric: float) -> bool:
        """Update state with the latest *metric* value.

        Args:
            metric: Latest validation metric.

        Returns:
            ``True`` when training should stop.
        """
        if self.best is None:
            self.best = metric
            return False

        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self) -> None:
        """Reset internal state."""
        self.best = None
        self.counter = 0
        self.should_stop = False


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: "nn.Module",
    loader: "DataLoader",
    optimizer: "torch.optim.Optimizer",
    criterion: "nn.Module",
    device: str = "cpu",
) -> Tuple[float, float]:
    """Run one training epoch.

    Args:
        model: PyTorch model.
        loader: Training data loader.
        optimizer: Gradient-based optimiser.
        criterion: Loss function.
        device: Torch device string.

    Returns:
        Tuple ``(mean_loss, accuracy)`` over the epoch.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for train_epoch.")

    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).long()
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct += (logits.detach().argmax(1) == y_batch).sum().item()
        n += len(y_batch)

    return total_loss / n, correct / n


def validate_epoch(
    model: "nn.Module",
    loader: "DataLoader",
    criterion: "nn.Module",
    device: str = "cpu",
) -> Tuple[float, float]:
    """Run one validation epoch.

    Args:
        model: PyTorch model.
        loader: Validation data loader.
        criterion: Loss function.
        device: Torch device string.

    Returns:
        Tuple ``(mean_loss, accuracy)`` over the epoch.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for validate_epoch.")

    import torch

    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).long()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            n += len(y_batch)

    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train_model(
    model: "nn.Module",
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    n_epochs: int = 100,
    lr: float = 1e-3,
    optimizer: str = "adam",
    scheduler: str = "cosine",
    patience: int = 15,
    checkpoint_dir: Optional[str | Path] = None,
    device: str = "cpu",
    model_name: str = "model",
    weight_decay: float = 1e-4,
) -> Dict:
    """Full training loop with early stopping, LR scheduling and checkpointing.

    Args:
        model: PyTorch model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        n_epochs: Maximum number of epochs.
        lr: Initial learning rate.
        optimizer: ``"adam"`` or ``"sgd"``.
        scheduler: ``"cosine"``, ``"step"``, or ``"none"``.
        patience: Early stopping patience in epochs.
        checkpoint_dir: Directory to save the best checkpoint.  ``None``
            disables checkpointing.
        device: Torch device string.
        model_name: Prefix for the checkpoint filename.
        weight_decay: L2 regularisation strength.

    Returns:
        History dictionary with keys ``"train_loss"``, ``"val_loss"``,
        ``"train_acc"``, ``"val_acc"``, ``"epoch_times"``.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for train_model.")

    import torch

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: '{optimizer}'")

    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    elif scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
    else:
        sched = None

    early_stop = EarlyStopping(patience=patience, mode="min")
    history: Dict = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "epoch_times": [],
    }
    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        t0 = time.time()
        t_loss, t_acc = train_epoch(model, train_loader, opt, criterion, device)
        v_loss, v_acc = validate_epoch(model, val_loader, criterion, device)
        epoch_time = time.time() - t0

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        history["epoch_times"].append(epoch_time)

        if sched is not None:
            sched.step()

        logger.info(
            "[%s] Epoch %d/%d | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f | %.1fs",
            model_name, epoch + 1, n_epochs,
            t_loss, t_acc, v_loss, v_acc, epoch_time,
        )

        # Checkpoint best model
        if v_loss < best_val_loss and checkpoint_dir is not None:
            best_val_loss = v_loss
            ckpt_path = Path(checkpoint_dir) / f"{model_name}_best.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "val_loss": v_loss,
                    "val_acc": v_acc,
                },
                ckpt_path,
            )
            logger.debug("Checkpoint saved to %s", ckpt_path)

        if early_stop(v_loss):
            logger.info("[%s] Early stopping at epoch %d.", model_name, epoch + 1)
            break

    return history


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    model: "nn.Module",
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> "nn.Module":
    """Load model weights from a checkpoint file.

    Args:
        model: PyTorch model instance (architecture must match checkpoint).
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        device: Torch device string.

    Returns:
        Model with loaded weights.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for load_checkpoint.")

    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            "Loaded checkpoint from %s (epoch %d, val_loss=%.4f)",
            checkpoint_path,
            checkpoint.get("epoch", -1),
            checkpoint.get("val_loss", float("nan")),
        )
    else:
        model.load_state_dict(checkpoint)
    return model.to(device)


# ---------------------------------------------------------------------------
# Learning curve utilities
# ---------------------------------------------------------------------------

def sample_training_fraction(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fraction: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified sub-sampling of the training set.

    Args:
        X_train: Feature array.
        y_train: Label array.
        fraction: Fraction of training samples to keep ``(0, 1]``.
        seed: Random seed.

    Returns:
        Tuple ``(X_sub, y_sub)``.
    """
    from sklearn.model_selection import train_test_split

    if fraction >= 1.0:
        return X_train, y_train

    _, X_sub, _, y_sub = train_test_split(
        X_train,
        y_train,
        test_size=fraction,
        stratify=y_train,
        random_state=seed,
    )
    return X_sub, y_sub


def run_learning_curve_experiment(
    model_builder: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fractions: Optional[List[float]] = None,
    n_repeats: int = 3,
    seeds: Optional[List[int]] = None,
    batch_size: int = 32,
    **train_kwargs,
) -> List[Dict]:
    """Run a learning curve experiment over different training-set fractions.

    For each fraction and seed, builds a fresh model, trains it, and records
    validation accuracy.

    Args:
        model_builder: Callable ``() → nn.Module`` that returns an untrained
            model.
        X_train: Full training features.
        y_train: Full training labels.
        X_val: Validation features.
        y_val: Validation labels.
        fractions: List of fractions to evaluate.
        n_repeats: Number of random seeds to average over.
        seeds: List of seeds (length >= ``n_repeats``).  Generated if
            ``None``.
        batch_size: Mini-batch size for data loaders.
        **train_kwargs: Additional keyword arguments forwarded to
            :func:`train_model`.

    Returns:
        List of result dicts, one per ``(fraction, seed)`` combination.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for run_learning_curve_experiment.")

    if fractions is None:
        fractions = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    if seeds is None:
        seeds = list(range(42, 42 + n_repeats))

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    val_ds = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    results = []
    for frac in fractions:
        for seed in seeds[:n_repeats]:
            X_sub, y_sub = sample_training_fraction(X_train, y_train, frac, seed=seed)
            X_sub_t = torch.tensor(X_sub, dtype=torch.float32)
            y_sub_t = torch.tensor(y_sub, dtype=torch.long)
            train_ds = TensorDataset(X_sub_t, y_sub_t)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            model = model_builder()
            history = train_model(
                model, train_loader, val_loader, **train_kwargs
            )
            best_val_acc = max(history["val_acc"])
            results.append({
                "fraction": frac,
                "seed": seed,
                "n_train": len(X_sub),
                "best_val_acc": best_val_acc,
                "history": history,
            })
            logger.info(
                "fraction=%.2f seed=%d n_train=%d best_val_acc=%.4f",
                frac, seed, len(X_sub), best_val_acc,
            )

    return results


def aggregate_repeated_runs(results_list: List[Dict]) -> Dict:
    """Aggregate learning curve results across seeds (mean ± std).

    Args:
        results_list: Output of :func:`run_learning_curve_experiment`.

    Returns:
        Dictionary ``{fraction: {"mean": float, "std": float, "n_train": int}}``.
    """
    from collections import defaultdict

    grouped: Dict[float, List[float]] = defaultdict(list)
    n_train_map: Dict[float, int] = {}
    for r in results_list:
        frac = r["fraction"]
        grouped[frac].append(r["best_val_acc"])
        n_train_map[frac] = r["n_train"]

    aggregated = {}
    for frac, accs in sorted(grouped.items()):
        aggregated[frac] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "n_train": n_train_map[frac],
        }
    return aggregated
