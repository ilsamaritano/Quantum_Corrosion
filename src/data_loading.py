"""IQ radar data loading, validation, normalisation, windowing and splitting."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_iq(
    data_dir: str | Path,
    file_pattern: str = "*.npy",
    class_mapping: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """Load raw IQ signals from numpy files stored in *data_dir*.

    Each ``.npy`` file should contain a 2-D array of shape ``(n_samples,
    signal_length)`` with complex or float values.  The file stem is used as
    the class name unless overridden by *class_mapping*.

    Args:
        data_dir: Directory that contains the ``.npy`` files.
        file_pattern: Glob pattern used to discover files.
        class_mapping: Optional ``{filename_stem: class_name}`` mapping.

    Returns:
        Dictionary ``{class_name: np.ndarray}`` where each array has shape
        ``(n_samples, signal_length)``.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted(data_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{file_pattern}' in {data_dir}"
        )

    samples: Dict[str, np.ndarray] = {}
    for fp in files:
        class_name = class_mapping.get(fp.stem, fp.stem) if class_mapping else fp.stem
        arr = np.load(fp, allow_pickle=False)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        samples[class_name] = arr
        logger.info("Loaded class '%s': %s samples", class_name, len(arr))

    return samples


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_samples(
    samples_dict: Dict[str, np.ndarray],
    min_length: int = 64,
    max_nan_ratio: float = 0.1,
) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    """Validate IQ samples: check shapes, NaN/Inf presence and minimum length.

    Args:
        samples_dict: ``{class_name: array}`` as returned by
            :func:`load_raw_iq`.
        min_length: Minimum required signal length (last dimension).
        max_nan_ratio: Maximum tolerated NaN/Inf ratio per sample.

    Returns:
        Tuple of ``(valid_dict, report_dict)`` where *report_dict* contains
        per-class validation statistics.
    """
    valid: Dict[str, np.ndarray] = {}
    report: Dict[str, dict] = {}

    for class_name, arr in samples_dict.items():
        n_total = len(arr)
        mask_valid = np.ones(n_total, dtype=bool)

        # Shape check — require 2-D
        if arr.ndim != 2:
            logger.warning(
                "Class '%s': expected 2-D array, got shape %s — skipping all.",
                class_name,
                arr.shape,
            )
            report[class_name] = {
                "n_total": n_total,
                "n_valid": 0,
                "reasons": ["wrong_ndim"],
            }
            continue

        # Minimum length check
        if arr.shape[1] < min_length:
            logger.warning(
                "Class '%s': signal length %d < min_length %d — skipping all.",
                class_name,
                arr.shape[1],
                min_length,
            )
            report[class_name] = {
                "n_total": n_total,
                "n_valid": 0,
                "reasons": ["too_short"],
            }
            continue

        # NaN / Inf per sample
        nan_inf_ratio = (np.isnan(arr) | np.isinf(arr)).mean(axis=1)
        bad_mask = nan_inf_ratio > max_nan_ratio
        mask_valid &= ~bad_mask

        n_valid = int(mask_valid.sum())
        report[class_name] = {
            "n_total": n_total,
            "n_valid": n_valid,
            "n_nan_inf_removed": int(bad_mask.sum()),
            "max_nan_ratio_found": float(nan_inf_ratio.max()),
        }

        if n_valid > 0:
            valid[class_name] = arr[mask_valid]

        logger.info(
            "Class '%s': %d/%d samples valid.",
            class_name,
            n_valid,
            n_total,
        )

    return valid, report


def remove_invalid_records(
    samples_dict: Dict[str, np.ndarray],
    validation_report: Dict[str, dict],
) -> Dict[str, np.ndarray]:
    """Remove classes where no valid samples remain according to *validation_report*.

    Args:
        samples_dict: Raw sample dictionary.
        validation_report: Report as produced by :func:`validate_samples`.

    Returns:
        Filtered dictionary with only classes that have at least one valid
        sample.
    """
    cleaned: Dict[str, np.ndarray] = {}
    for class_name, arr in samples_dict.items():
        rep = validation_report.get(class_name, {})
        if rep.get("n_valid", len(arr)) > 0:
            cleaned[class_name] = arr
        else:
            logger.warning("Dropping class '%s': no valid samples.", class_name)
    return cleaned


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_signal(
    signal: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    """Normalise an IQ signal array.

    Args:
        signal: 1-D or 2-D float array (samples last if 2-D).
        method: One of ``"zscore"``, ``"minmax"``, ``"unit_norm"``.

    Returns:
        Normalised array of the same shape.
    """
    signal = np.asarray(signal, dtype=np.float64)

    if method == "zscore":
        mean = signal.mean(axis=-1, keepdims=True)
        std = signal.std(axis=-1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (signal - mean) / std

    if method == "minmax":
        mn = signal.min(axis=-1, keepdims=True)
        mx = signal.max(axis=-1, keepdims=True)
        denom = np.where(mx - mn == 0, 1.0, mx - mn)
        return (signal - mn) / denom

    if method == "unit_norm":
        norm = np.linalg.norm(signal, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        return signal / norm

    raise ValueError(f"Unknown normalisation method: '{method}'")


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def window_signal(
    signal: np.ndarray,
    window_size: int = 256,
    hop_size: int = 128,
    window_fn: str = "hann",
) -> np.ndarray:
    """Segment a 1-D IQ signal into overlapping windows.

    Args:
        signal: 1-D float array.
        window_size: Number of samples per window.
        hop_size: Step between successive windows.
        window_fn: Window function name — ``"hann"``, ``"hamming"``,
            ``"blackman"``, or ``"rectangular"``.

    Returns:
        2-D array of shape ``(n_windows, window_size)``.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = len(signal)

    # Build window weights
    if window_fn == "hann":
        window = np.hanning(window_size)
    elif window_fn == "hamming":
        window = np.hamming(window_size)
    elif window_fn == "blackman":
        window = np.blackman(window_size)
    else:
        window = np.ones(window_size)

    starts = list(range(0, n - window_size + 1, hop_size))
    if not starts:
        return np.empty((0, window_size), dtype=np.float64)
    segments = np.stack(
        [signal[s : s + window_size] * window for s in starts]
    )
    return segments


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def load_or_generate_synthetic(
    n_classes: int = 4,
    n_samples_per_class: int = 200,
    signal_length: int = 512,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate synthetic IQ radar data suitable for testing the pipeline.

    Each class has a distinct centre frequency and modulation depth so that a
    classifier can separate them.  Real and imaginary components are returned
    interleaved as a single float array of length ``signal_length * 2``.

    Args:
        n_classes: Number of corrosion severity classes.
        n_samples_per_class: Samples generated per class.
        signal_length: Length of each IQ signal.
        seed: Random seed.

    Returns:
        Dictionary ``{class_name: array}`` with shape
        ``(n_samples_per_class, signal_length)``.
    """
    rng = np.random.default_rng(seed)
    class_names = [
        "healthy",
        "light_corrosion",
        "moderate_corrosion",
        "severe_corrosion",
    ][:n_classes]

    t = np.linspace(0, 1, signal_length)
    base_freqs = [5, 15, 30, 50][:n_classes]
    noise_scales = [0.05, 0.10, 0.20, 0.40][:n_classes]

    samples: Dict[str, np.ndarray] = {}
    for i, name in enumerate(class_names):
        f0 = base_freqs[i]
        noise_std = noise_scales[i]
        batch = []
        for _ in range(n_samples_per_class):
            # Carrier with slight per-sample frequency jitter
            f_jitter = f0 + rng.normal(0, 0.5)
            phase = rng.uniform(0, 2 * np.pi)
            carrier = np.sin(2 * np.pi * f_jitter * t + phase)
            # Add harmonic to make classes more distinct
            harmonic = 0.3 * np.sin(2 * np.pi * 2 * f_jitter * t + phase)
            # Amplitude modulation
            am = 1 + 0.2 * np.sin(2 * np.pi * 2 * t)
            signal = am * (carrier + harmonic) + rng.normal(0, noise_std, signal_length)
            batch.append(signal)
        samples[name] = np.array(batch, dtype=np.float32)

    return samples


# ---------------------------------------------------------------------------
# Splitting and persistence
# ---------------------------------------------------------------------------

def save_processed_dataset(
    data: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> None:
    """Stratified split and save train / val / test arrays to *output_dir*.

    Args:
        data: Feature array of shape ``(n_samples, ...)``.
        labels: Integer label array of shape ``(n_samples,)``.
        output_dir: Directory where split ``.npy`` files are written.
        split_ratios: ``(train, val, test)`` fractions that must sum to 1.
        seed: Random seed for splitting.
    """
    from sklearn.model_selection import train_test_split  # local import

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ratio, val_ratio, test_ratio = split_ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "split_ratios must sum to 1"

    # First split: (train+val) vs test
    X_tv, X_test, y_tv, y_test = train_test_split(
        data, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )
    # Second split: train vs val
    relative_val = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=relative_val,
        stratify=y_tv,
        random_state=seed,
    )

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)

    logger.info(
        "Saved splits → train: %d, val: %d, test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )


def load_splits(
    splits_dir: str | Path,
) -> Dict[str, np.ndarray]:
    """Load train / val / test splits from *splits_dir*.

    Args:
        splits_dir: Directory containing ``X_train.npy``, ``y_train.npy``,
            etc.

    Returns:
        Dictionary with keys ``X_train``, ``y_train``, ``X_val``, ``y_val``,
        ``X_test``, ``y_test``.
    """
    splits_dir = Path(splits_dir)
    keys = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    result: Dict[str, np.ndarray] = {}
    for k in keys:
        fp = splits_dir / f"{k}.npy"
        if not fp.exists():
            raise FileNotFoundError(f"Split file not found: {fp}")
        result[k] = np.load(fp, allow_pickle=False)
    logger.info(
        "Loaded splits from %s — train: %d, val: %d, test: %d",
        splits_dir,
        len(result["X_train"]),
        len(result["X_val"]),
        len(result["X_test"]),
    )
    return result
