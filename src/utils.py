"""Shared utility helpers for the QuantumCorrosion pipeline."""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy, torch, and Python random.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device() -> "torch.device":
    """Return the best available torch device (CUDA if available, else CPU).

    Returns:
        torch.device instance.
    """
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("Using device: %s", device)
    return device


def save_json(data: Any, path: str | Path) -> None:
    """Serialise *data* to a JSON file at *path*.

    Args:
        data: JSON-serialisable object.
        path: Destination file path.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)
    logger.debug("Saved JSON to %s", path)


def load_json(path: str | Path) -> Any:
    """Load a JSON file and return the parsed object.

    Args:
        path: Source file path.

    Returns:
        Parsed JSON object.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and all parents) if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string (seconds precision).

    Returns:
        Timestamp string, e.g. ``"2024-01-15T12:34:56"``.
    """
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger with a consistent format.

    Args:
        log_level: One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
        log_file: Optional path to write log output to a file in addition to
            stdout.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        ensure_dir(Path(log_file).parent)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logger.debug("Logging initialised at level %s", log_level)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Fallback serialiser for types not natively supported by json.dump."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
