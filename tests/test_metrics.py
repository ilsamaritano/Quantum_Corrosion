"""Tests for metrics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    compute_classification_metrics,
    compute_per_class_metrics,
    load_metrics_csv,
    save_metrics_csv,
    summarize_model_comparison,
)


# ---------------------------------------------------------------------------
# compute_classification_metrics
# ---------------------------------------------------------------------------

class TestComputeClassificationMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 3] * 5)
        metrics = compute_classification_metrics(y, y)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y_true = np.zeros(20, dtype=int)
        y_pred = np.ones(20, dtype=int)
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(0.0)

    def test_binary_case(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        metrics = compute_classification_metrics(y_true, y_pred, average="binary")
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_returns_required_keys(self):
        y = np.array([0, 1, 0, 1])
        metrics = compute_classification_metrics(y, y)
        for key in ["accuracy", "f1", "precision", "recall", "balanced_accuracy"]:
            assert key in metrics

    def test_with_probabilities(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        metrics = compute_classification_metrics(y_true, y_pred, y_proba=proba)
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] == pytest.approx(1.0)

    def test_four_class(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 4, size=100)
        y_pred = rng.integers(0, 4, size=100)
        metrics = compute_classification_metrics(y_true, y_pred)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0


# ---------------------------------------------------------------------------
# compute_per_class_metrics
# ---------------------------------------------------------------------------

class TestComputePerClassMetrics:
    def test_returns_dataframe(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        df = compute_per_class_metrics(y, y)
        assert isinstance(df, pd.DataFrame)

    def test_number_of_rows(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        df = compute_per_class_metrics(y, y)
        assert len(df) == 3

    def test_with_class_names(self):
        y = np.array([0, 1, 0, 1])
        df = compute_per_class_metrics(y, y, class_names=["cat", "dog"])
        assert "cat" in df.index
        assert "dog" in df.index

    def test_perfect_precision_recall(self):
        y = np.array([0, 1, 2] * 10)
        df = compute_per_class_metrics(y, y, class_names=["a", "b", "c"])
        assert df.loc["a", "precision"] == pytest.approx(1.0)
        assert df.loc["a", "recall"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# summarize_model_comparison
# ---------------------------------------------------------------------------

class TestSummarizeModelComparison:
    def test_returns_dataframe(self):
        results = {
            "model_a": {"accuracy": 0.9, "f1": 0.89},
            "model_b": {"accuracy": 0.85, "f1": 0.84},
        }
        df = summarize_model_comparison(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_index_is_model_name(self):
        results = {"alpha": {"accuracy": 0.5}, "beta": {"accuracy": 0.7}}
        df = summarize_model_comparison(results)
        assert "alpha" in df.index
        assert "beta" in df.index

    def test_columns_ordered(self):
        results = {"m": {"accuracy": 0.8, "f1": 0.79, "precision": 0.80,
                         "recall": 0.78, "n_params": 1000}}
        df = summarize_model_comparison(results, include_params=True)
        assert "accuracy" in df.columns


# ---------------------------------------------------------------------------
# save_metrics_csv / load_metrics_csv
# ---------------------------------------------------------------------------

class TestSaveLoadMetricsCsv:
    def test_roundtrip(self, tmp_path):
        data = {
            "model_a": {"accuracy": 0.9, "f1": 0.88},
            "model_b": {"accuracy": 0.8, "f1": 0.79},
        }
        df = pd.DataFrame(data).T
        df.index.name = "model"
        path = tmp_path / "metrics.csv"
        save_metrics_csv(df, path)
        loaded = load_metrics_csv(path)
        assert loaded.shape == df.shape
        assert list(loaded.columns) == list(df.columns)

    def test_values_preserved(self, tmp_path):
        df = pd.DataFrame(
            {"accuracy": [0.95, 0.88], "f1": [0.94, 0.87]},
            index=pd.Index(["a", "b"], name="model"),
        )
        path = tmp_path / "test_metrics.csv"
        save_metrics_csv(df, path)
        loaded = load_metrics_csv(path)
        assert loaded.loc["a", "accuracy"] == pytest.approx(0.95)
        assert loaded.loc["b", "f1"] == pytest.approx(0.87)

    def test_creates_parent_dir(self, tmp_path):
        df = pd.DataFrame({"acc": [0.9]}, index=pd.Index(["x"], name="model"))
        path = tmp_path / "sub" / "dir" / "m.csv"
        save_metrics_csv(df, path)
        assert path.exists()
