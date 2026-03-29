#!/usr/bin/env python3
"""Data preparation script: load (or generate synthetic) IQ radar data,
validate, preprocess to STFT spectrograms, and save train/val/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from src.data_loading import (
    load_or_generate_synthetic,
    load_raw_iq,
    remove_invalid_records,
    save_processed_dataset,
    validate_samples,
)
from src.preprocessing import batch_preprocess, extract_tabular_features
from src.spectrograms import resize_spectrogram
from src.utils import ensure_dir, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare IQ radar dataset for training."
    )
    parser.add_argument(
        "--data_dir",
        default="data/raw",
        help="Directory containing raw .npy IQ files.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/splits",
        help="Directory where train/val/test splits are saved.",
    )
    parser.add_argument(
        "--processed_dir",
        default="data/processed",
        help="Directory where processed spectrogram arrays are saved.",
    )
    parser.add_argument(
        "--config",
        default="configs/dataset.yaml",
        help="Path to dataset configuration YAML.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="Force synthetic data generation even if raw files exist.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    # Load config
    cfg: dict = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)

    ds_cfg = cfg.get("dataset", {})
    n_classes = ds_cfg.get("n_classes", 4)
    n_samples = ds_cfg.get("n_samples_per_class", 200)
    signal_length = ds_cfg.get("signal_length", 512)
    class_names = ds_cfg.get(
        "class_names",
        ["healthy", "light_corrosion", "moderate_corrosion", "severe_corrosion"],
    )

    pp_cfg = cfg.get("preprocessing", {})
    n_fft = pp_cfg.get("n_fft", 256)
    hop_length = pp_cfg.get("hop_length", 64)
    target_size = tuple(pp_cfg.get("target_size", [64, 64]))

    split_cfg = cfg.get("splits", {})
    train_r = split_cfg.get("train", 0.7)
    val_r = split_cfg.get("val", 0.15)
    test_r = split_cfg.get("test", 0.15)

    # -----------------------------------------------------------------------
    # Load or generate data
    # -----------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    force_synthetic = args.synthetic or ds_cfg.get("synthetic", False)

    if not force_synthetic and any(data_dir.glob("*.npy")):
        print(f"Loading real IQ data from {data_dir} …")
        raw = load_raw_iq(data_dir)
    else:
        print("Generating synthetic IQ data …")
        raw = load_or_generate_synthetic(
            n_classes=n_classes,
            n_samples_per_class=n_samples,
            signal_length=signal_length,
            seed=args.seed,
        )

    # -----------------------------------------------------------------------
    # Validate
    # -----------------------------------------------------------------------
    valid, report = validate_samples(raw)
    valid = remove_invalid_records(valid, report)

    print("\n=== Validation Report ===")
    for cls, rep in report.items():
        print(f"  {cls}: {rep}")

    # -----------------------------------------------------------------------
    # Print data statistics
    # -----------------------------------------------------------------------
    print("\n=== Data Statistics ===")
    total = 0
    for cname, arr in valid.items():
        print(
            f"  {cname}: {len(arr)} samples, signal_length={arr.shape[1]}, "
            f"mean={arr.mean():.4f}, std={arr.std():.4f}"
        )
        total += len(arr)
    print(f"  Total: {total} samples across {len(valid)} classes")

    # -----------------------------------------------------------------------
    # Flatten to arrays
    # -----------------------------------------------------------------------
    all_signals = []
    all_labels = []
    names_sorted = sorted(valid.keys())
    name_to_idx = {n: i for i, n in enumerate(names_sorted)}

    for cname in names_sorted:
        arr = valid[cname]
        all_signals.append(arr)
        all_labels.append(np.full(len(arr), name_to_idx[cname], dtype=np.int64))

    X_raw = np.concatenate(all_signals, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # -----------------------------------------------------------------------
    # Preprocess to spectrograms and resize
    # -----------------------------------------------------------------------
    print(f"\nPreprocessing {len(X_raw)} signals to spectrograms …")
    specs = batch_preprocess(X_raw, n_fft=n_fft, hop_length=hop_length)

    # Resize to uniform target_size
    specs_resized = np.stack(
        [resize_spectrogram(s, target_size=target_size) for s in specs]
    )
    print(f"Spectrogram array shape: {specs_resized.shape}")

    # Save processed array
    processed_dir = Path(args.processed_dir)
    ensure_dir(processed_dir)
    np.save(processed_dir / "spectrograms.npy", specs_resized)
    np.save(processed_dir / "labels.npy", y)
    print(f"Saved processed spectrograms to {processed_dir}")

    # -----------------------------------------------------------------------
    # Tabular features (for quantum / MLP models)
    # -----------------------------------------------------------------------
    print("Extracting tabular features …")
    tab_features = np.stack([extract_tabular_features(s) for s in X_raw])
    np.save(processed_dir / "tabular_features.npy", tab_features)
    print(f"Tabular features shape: {tab_features.shape}")

    # -----------------------------------------------------------------------
    # Save splits
    # -----------------------------------------------------------------------
    print(f"\nSaving train/val/test splits to {args.output_dir} …")
    save_processed_dataset(
        specs_resized,
        y,
        output_dir=args.output_dir,
        split_ratios=(train_r, val_r, test_r),
        seed=args.seed,
    )

    # Also save tabular splits
    tab_splits_dir = Path(args.output_dir) / "tabular"
    save_processed_dataset(
        tab_features,
        y,
        output_dir=tab_splits_dir,
        split_ratios=(train_r, val_r, test_r),
        seed=args.seed,
    )

    print("\nData preparation complete.")
    print(f"  Label mapping: {name_to_idx}")


if __name__ == "__main__":
    main()
