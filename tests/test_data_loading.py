"""Tests for data_loading module (uses synthetic data only)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data_loading import (
    load_or_generate_synthetic,
    load_splits,
    normalize_signal,
    remove_invalid_records,
    save_processed_dataset,
    validate_samples,
    window_signal,
)


# ---------------------------------------------------------------------------
# load_or_generate_synthetic
# ---------------------------------------------------------------------------

class TestLoadOrGenerateSynthetic:
    def test_returns_dict(self):
        data = load_or_generate_synthetic(n_classes=4, n_samples_per_class=10, signal_length=64)
        assert isinstance(data, dict)
        assert len(data) == 4

    def test_class_names(self):
        data = load_or_generate_synthetic(n_classes=2, n_samples_per_class=5)
        assert "healthy" in data
        assert "light_corrosion" in data

    def test_array_shape(self):
        data = load_or_generate_synthetic(n_classes=3, n_samples_per_class=8, signal_length=128)
        for arr in data.values():
            assert arr.ndim == 2
            assert arr.shape == (8, 128)

    def test_reproducibility(self):
        a = load_or_generate_synthetic(seed=42)
        b = load_or_generate_synthetic(seed=42)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_different_seeds(self):
        a = load_or_generate_synthetic(seed=1)
        b = load_or_generate_synthetic(seed=2)
        first_key = list(a.keys())[0]
        assert not np.array_equal(a[first_key], b[first_key])


# ---------------------------------------------------------------------------
# validate_samples
# ---------------------------------------------------------------------------

class TestValidateSamples:
    def test_valid_samples_pass(self):
        data = load_or_generate_synthetic(n_classes=2, n_samples_per_class=10)
        valid, report = validate_samples(data, min_length=64)
        for cname, rep in report.items():
            assert rep["n_valid"] == 10

    def test_nan_removal(self):
        arr = np.ones((20, 128), dtype=np.float32)
        arr[5, :] = np.nan  # entire sample is NaN
        data = {"test_class": arr}
        valid, report = validate_samples(data, max_nan_ratio=0.0)
        assert report["test_class"]["n_nan_inf_removed"] == 1

    def test_short_signal_rejected(self):
        arr = np.ones((10, 16), dtype=np.float32)
        data = {"short_class": arr}
        _, report = validate_samples(data, min_length=64)
        assert report["short_class"]["n_valid"] == 0

    def test_returns_two_dicts(self):
        data = load_or_generate_synthetic(n_classes=2, n_samples_per_class=5)
        result = validate_samples(data)
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert isinstance(result[1], dict)


# ---------------------------------------------------------------------------
# remove_invalid_records
# ---------------------------------------------------------------------------

class TestRemoveInvalidRecords:
    def test_removes_zero_valid_class(self):
        data = load_or_generate_synthetic(n_classes=2, n_samples_per_class=5)
        names = list(data.keys())
        report = {names[0]: {"n_valid": 5}, names[1]: {"n_valid": 0}}
        cleaned = remove_invalid_records(data, report)
        assert names[0] in cleaned
        assert names[1] not in cleaned

    def test_keeps_all_valid(self):
        data = load_or_generate_synthetic(n_classes=3, n_samples_per_class=5)
        report = {n: {"n_valid": 5} for n in data}
        cleaned = remove_invalid_records(data, report)
        assert len(cleaned) == 3


# ---------------------------------------------------------------------------
# normalize_signal
# ---------------------------------------------------------------------------

class TestNormalizeSignal:
    def test_zscore_zero_mean(self):
        sig = np.random.randn(256).astype(np.float32)
        norm = normalize_signal(sig, method="zscore")
        assert abs(norm.mean()) < 1e-5

    def test_zscore_unit_std(self):
        sig = np.random.randn(256).astype(np.float32) * 10 + 5
        norm = normalize_signal(sig, method="zscore")
        assert abs(norm.std() - 1.0) < 1e-4

    def test_minmax_range(self):
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        norm = normalize_signal(sig, method="minmax")
        assert norm.min() == pytest.approx(0.0)
        assert norm.max() == pytest.approx(1.0)

    def test_unit_norm(self):
        sig = np.array([3.0, 4.0])
        norm = normalize_signal(sig, method="unit_norm")
        assert np.linalg.norm(norm) == pytest.approx(1.0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            normalize_signal(np.ones(10), method="bogus")


# ---------------------------------------------------------------------------
# window_signal
# ---------------------------------------------------------------------------

class TestWindowSignal:
    def test_output_shape(self):
        sig = np.random.randn(512)
        windows = window_signal(sig, window_size=256, hop_size=128)
        assert windows.ndim == 2
        assert windows.shape[1] == 256
        # At least 2 windows
        assert windows.shape[0] >= 2

    def test_rectangular_window(self):
        sig = np.ones(512)
        windows = window_signal(sig, window_size=128, hop_size=64, window_fn="rectangular")
        np.testing.assert_allclose(windows[0], np.ones(128))

    def test_no_windows_for_short_signal(self):
        sig = np.ones(10)
        windows = window_signal(sig, window_size=128, hop_size=64)
        # Signal shorter than window — no full windows
        assert windows.shape[0] == 0


# ---------------------------------------------------------------------------
# save_processed_dataset / load_splits
# ---------------------------------------------------------------------------

class TestSaveLoadSplits:
    def test_roundtrip(self, tmp_path):
        X = np.random.randn(100, 16).astype(np.float32)
        y = np.repeat(np.arange(4), 25).astype(np.int64)
        save_processed_dataset(X, y, tmp_path, split_ratios=(0.7, 0.15, 0.15), seed=0)
        splits = load_splits(tmp_path)
        assert splits["X_train"].shape[1] == 16
        n_total = len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"])
        assert n_total == 100

    def test_stratified_split(self, tmp_path):
        X = np.random.randn(40, 8).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10, dtype=np.int64)
        save_processed_dataset(X, y, tmp_path, seed=42)
        splits = load_splits(tmp_path)
        # Each split should contain all classes
        for part in ["y_train", "y_val", "y_test"]:
            assert len(np.unique(splits[part])) == 4
