"""Tests for model building and forward passes (classical and quantum)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.classical_models import (
    build_mlp_classifier,
    build_mobilenetv2,
    build_resnet34,
    build_simple_cnn,
    count_trainable_params,
    get_model_summary,
)


# ---------------------------------------------------------------------------
# SimpleCNN
# ---------------------------------------------------------------------------

class TestBuildSimpleCNN:
    def test_build(self):
        model = build_simple_cnn(num_classes=4, input_channels=1)
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self):
        model = build_simple_cnn(num_classes=4, input_channels=1)
        model.eval()
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 4)

    def test_three_channel_input(self):
        model = build_simple_cnn(num_classes=3, input_channels=3)
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 3)

    def test_param_count_positive(self):
        model = build_simple_cnn(num_classes=4)
        assert count_trainable_params(model) > 0


# ---------------------------------------------------------------------------
# MLP classifier
# ---------------------------------------------------------------------------

class TestBuildMlpClassifier:
    def test_build(self):
        model = build_mlp_classifier(input_dim=16, num_classes=4)
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self):
        model = build_mlp_classifier(input_dim=16, num_classes=4)
        model.eval()
        x = torch.randn(8, 16)
        out = model(x)
        assert out.shape == (8, 4)

    def test_custom_hidden(self):
        model = build_mlp_classifier(input_dim=32, num_classes=3, hidden_dims=[64, 32, 16])
        model.eval()
        x = torch.randn(4, 32)
        out = model(x)
        assert out.shape == (4, 3)


# ---------------------------------------------------------------------------
# ResNet-34
# ---------------------------------------------------------------------------

class TestBuildResnet34:
    def test_build_no_pretrained(self):
        model = build_resnet34(num_classes=4, pretrained=False)
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self):
        model = build_resnet34(num_classes=4, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 4)

    def test_single_channel_input(self):
        model = build_resnet34(num_classes=4, pretrained=False, input_channels=1)
        model.eval()
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 4)

    def test_param_count(self):
        model = build_resnet34(num_classes=4, pretrained=False)
        n = count_trainable_params(model)
        # ResNet-34 has ~21M params for ImageNet, with 4 classes will be slightly less
        assert n > 1_000_000


# ---------------------------------------------------------------------------
# MobileNetV2
# ---------------------------------------------------------------------------

class TestBuildMobilenetV2:
    def test_build(self):
        model = build_mobilenetv2(num_classes=4, pretrained=False)
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self):
        model = build_mobilenetv2(num_classes=4, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# count_trainable_params
# ---------------------------------------------------------------------------

class TestCountTrainableParams:
    def test_frozen_params_excluded(self):
        model = build_simple_cnn(num_classes=4)
        for p in model.parameters():
            p.requires_grad = False
        assert count_trainable_params(model) == 0

    def test_returns_int(self):
        model = build_mlp_classifier(16, 4)
        n = count_trainable_params(model)
        assert isinstance(n, int)


# ---------------------------------------------------------------------------
# get_model_summary
# ---------------------------------------------------------------------------

class TestGetModelSummary:
    def test_returns_dict(self):
        model = build_simple_cnn(num_classes=4)
        summary = get_model_summary(model, input_size=(1, 1, 64, 64))
        assert isinstance(summary, dict)
        assert "total_params" in summary
        assert "trainable_params" in summary
        assert "layers" in summary

    def test_trainable_params_consistent(self):
        model = build_mlp_classifier(16, 4)
        summary = get_model_summary(model, input_size=(1, 16))
        assert summary["trainable_params"] == count_trainable_params(model)


# ---------------------------------------------------------------------------
# HybridQuantumClassifier (requires PennyLane)
# ---------------------------------------------------------------------------

try:
    import pennylane  # noqa: F401

    _PENNYLANE_AVAILABLE = True
except ImportError:
    _PENNYLANE_AVAILABLE = False


@pytest.mark.skipif(not _PENNYLANE_AVAILABLE, reason="PennyLane not installed")
class TestHybridQuantumClassifier:
    def test_build(self):
        from src.quantum_models import HybridQuantumClassifier

        model = HybridQuantumClassifier(
            input_dim=8, n_qubits=2, n_layers=1, n_classes=2
        )
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self):
        from src.quantum_models import HybridQuantumClassifier

        model = HybridQuantumClassifier(
            input_dim=8, n_qubits=2, n_layers=1, n_classes=2
        )
        model.eval()
        x = torch.randn(3, 8)
        out = model(x)
        assert out.shape == (3, 2)

    def test_param_count(self):
        from src.quantum_models import HybridQuantumClassifier

        model = HybridQuantumClassifier(
            input_dim=4, n_qubits=2, n_layers=1, n_classes=2
        )
        n = count_trainable_params(model)
        assert n > 0
