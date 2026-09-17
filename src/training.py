"""
Training Engine — Optimised training loop
==========================================
Supports AMP, gradient clipping, cosine annealing, early stopping,
per-round reproducibility, and both classical & quantum models.
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field, asdict
import logging
import copy
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainMetrics:
    """Per-epoch metrics container."""
    epoch: int = 0
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    lr: float = 0.0
    epoch_time: float = 0.0


@dataclass
class RoundResult:
    """Full result of one training round."""
    round_id: int = 0
    best_epoch: int = 0
    best_val_acc: float = 0.0
    best_val_loss: float = float("inf")
    history: List[Dict] = field(default_factory=list)
    train_time_total: float = 0.0


def set_seed(seed: int):
    """Full deterministic seeding."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping with patience and best-model checkpoint."""

    def __init__(self, patience: int = 20, delta: float = 1e-5):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def _get_scheduler(optimizer, name: str, epochs: int):
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    elif name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)
    else:
        return None


# ── Core training loop ──────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: Optional[GradScaler] = None,
    grad_clip: float = 1.0,
    use_amp: bool = True,
    log_every_batches: int = 50,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, total=len(loader), desc="Train", leave=False, dynamic_ncols=True)
    for batch_idx, (inputs, labels) in enumerate(pbar, start=1):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None and device == "cuda":
            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if log_every_batches > 0 and (batch_idx % log_every_batches == 0):
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.4f}")
            logger.info(
                f"    batch {batch_idx}/{len(loader)} │ "
                f"loss {loss.item():.4f} │ acc {correct / max(total, 1):.4f}"
            )

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Evaluate on val/test set. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, total=len(loader), desc="Val", leave=False, dynamic_ncols=True)
    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.4f}")

    return total_loss / total, correct / total


# ── Full training pipeline ──────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 150,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    scheduler_name: str = "cosine",
    patience: int = 20,
    use_amp: bool = True,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.05,
    class_weights: Optional[torch.Tensor] = None,
    round_id: int = 0,
    save_dir: Optional[Path] = None,
    model_name: str = "model",
) -> RoundResult:
    """
    Full training pipeline for one round.
    Returns RoundResult with history and best checkpoint.
    """
    model = model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _get_scheduler(optimizer, scheduler_name, epochs)
    scaler = GradScaler(device="cuda") if (use_amp and device == "cuda") else None
    early_stop = EarlyStopping(patience=patience)

    result = RoundResult(round_id=round_id)
    t_start = time.time()

    epoch_iter = tqdm(range(1, epochs + 1), desc=f"Round {round_id}", dynamic_ncols=True)
    for epoch in epoch_iter:
        t_epoch = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, grad_clip, use_amp, log_every_batches=25,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        epoch_time = time.time() - t_epoch

        metrics = TrainMetrics(
            epoch=epoch, train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss, val_acc=val_acc, lr=current_lr,
            epoch_time=epoch_time,
        )
        result.history.append(asdict(metrics))
        epoch_iter.set_postfix(
            train_acc=f"{train_acc:.4f}",
            val_acc=f"{val_acc:.4f}",
            lr=f"{current_lr:.2e}",
        )

        if val_acc > result.best_val_acc:
            result.best_val_acc = val_acc
            result.best_val_loss = val_loss
            result.best_epoch = epoch

        early_stop(val_loss, model)

        if epoch % 25 == 0 or epoch == 1:
            logger.info(
                f"  [R{round_id}] Epoch {epoch:3d}/{epochs} │ "
                f"Train {train_loss:.4f}/{train_acc:.4f} │ "
                f"Val {val_loss:.4f}/{val_acc:.4f} │ "
                f"LR {current_lr:.2e} │ {epoch_time:.1f}s"
            )

        if early_stop.stop:
            logger.info(f"  [R{round_id}] Early stopping at epoch {epoch}")
            break

    # Restore best model
    if early_stop.best_state is not None:
        model.load_state_dict(early_stop.best_state)

    result.train_time_total = time.time() - t_start

    # Save checkpoint
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"{model_name}_round{round_id}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "round_id": round_id,
            "best_epoch": result.best_epoch,
            "best_val_acc": result.best_val_acc,
        }, ckpt_path)
        # Save history
        hist_path = save_dir / f"{model_name}_round{round_id}_history.json"
        with open(hist_path, 'w') as f:
            json.dump(result.history, f, indent=2)

    logger.info(
        f"  [R{round_id}] Done: best val acc={result.best_val_acc:.4f} "
        f"@ epoch {result.best_epoch}, total {result.train_time_total:.1f}s"
    )

    return result


# ── Multi-round training ────────────────────────────────────────────────────

def train_multi_round(
    model_factory,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_rounds: int = 10,
    base_seed: int = 42,
    **train_kwargs,
) -> List[RoundResult]:
    """
    Run `n_rounds` independent training rounds with different seeds.
    `model_factory` is a callable that returns a fresh model instance.
    """
    all_results = []
    for r in range(n_rounds):
        seed = base_seed + r
        set_seed(seed)
        logger.info(f"\n{'='*60}\nRound {r+1}/{n_rounds} (seed={seed})\n{'='*60}")

        model = model_factory()
        result = train_model(model, train_loader, val_loader,
                             round_id=r, **train_kwargs)
        all_results.append(result)

    # Summary statistics
    accs = [r.best_val_acc for r in all_results]
    logger.info(
        f"\n{'='*60}\n"
        f"Multi-round summary ({n_rounds} rounds):\n"
        f"  Mean val acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}\n"
        f"  Min: {np.min(accs):.4f}  Max: {np.max(accs):.4f}\n"
        f"  Variance: {np.var(accs):.6e}\n"
        f"{'='*60}"
    )
    return all_results
