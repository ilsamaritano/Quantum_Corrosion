"""Tests for preprocessing module."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import (
    augment_signal,
    batch_preprocess,
    compute_fft,
    extract_tabular_features,
    standardize_features,
)
from src.spectrograms import (
    mel_spectrogram,
    resize_spectrogram,
    spectrogram_to_tensor,
    stft_spectrogram,
)


# ---------------------------------------------------------------------------
# compute_fft
# ---------------------------------------------------------------------------

class TestComputeFFT:
    def test_magnitude_shape(self):
        sig = np.random.randn(512)
        mag = compute_fft(sig)
        assert mag.ndim == 1
        # Single-sided: n_fft // 2 + 1
        assert len(mag) == 512 // 2 + 1

    def test_non_negative_magnitude(self):
        sig = np.random.randn(256)
        mag = compute_fft(sig)
        assert np.all(mag >= 0)

    def test_complex_return(self):
        sig = np.random.randn(128)
        spec = compute_fft(sig, return_complex=True)
        assert np.iscomplexobj(spec)
        assert len(spec) == 128

    def test_custom_nfft(self):
        sig = np.random.randn(100)
        mag = compute_fft(sig, n_fft=256)
        assert len(mag) == 256 // 2 + 1


# ---------------------------------------------------------------------------
# stft_spectrogram
# ---------------------------------------------------------------------------

class TestStftSpectrogram:
    def test_output_shape(self):
        sig = np.random.randn(512)
        spec = stft_spectrogram(sig, n_fft=64, hop_length=32)
        assert spec.ndim == 2
        assert spec.shape[0] == 64 // 2 + 1

    def test_normalized_range(self):
        sig = np.random.randn(512)
        spec = stft_spectrogram(sig, normalize=True)
        assert spec.min() >= -0.01
        assert spec.max() <= 1.01

    def test_short_signal_pads(self):
        sig = np.random.randn(16)  # shorter than default n_fft=256
        spec = stft_spectrogram(sig, n_fft=256, hop_length=64)
        assert spec.ndim == 2

    def test_float32_output(self):
        sig = np.random.randn(256)
        spec = stft_spectrogram(sig)
        assert spec.dtype == np.float32


# ---------------------------------------------------------------------------
# mel_spectrogram
# ---------------------------------------------------------------------------

class TestMelSpectrogram:
    def test_output_shape(self):
        sig = np.random.randn(512)
        mel = mel_spectrogram(sig, n_mels=32, n_fft=64, hop_length=16)
        assert mel.shape[0] == 32
        assert mel.ndim == 2

    def test_normalized_range(self):
        sig = np.random.randn(512)
        mel = mel_spectrogram(sig, n_mels=16)
        assert mel.min() >= -0.01
        assert mel.max() <= 1.01


# ---------------------------------------------------------------------------
# resize_spectrogram
# ---------------------------------------------------------------------------

class TestResizeSpectrogram:
    def test_output_size(self):
        spec = np.random.rand(33, 15).astype(np.float32)
        resized = resize_spectrogram(spec, target_size=(64, 64))
        assert resized.shape == (64, 64)

    def test_square_input_stays_square(self):
        spec = np.eye(32, dtype=np.float32)
        resized = resize_spectrogram(spec, target_size=(32, 32))
        assert resized.shape == (32, 32)


# ---------------------------------------------------------------------------
# spectrogram_to_tensor
# ---------------------------------------------------------------------------

class TestSpectrogramToTensor:
    def test_output_shape(self):
        import torch

        spec = np.random.rand(32, 32).astype(np.float32)
        t = spectrogram_to_tensor(spec)
        assert t.shape == (3, 32, 32)
        assert isinstance(t, torch.Tensor)

    def test_normalized_range(self):
        spec = np.random.rand(16, 16).astype(np.float32)
        t = spectrogram_to_tensor(spec, normalize=True)
        assert t.min().item() >= 0.0
        assert t.max().item() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# standardize_features
# ---------------------------------------------------------------------------

class TestStandardizeFeatures:
    def test_zero_mean(self):
        X = np.random.randn(100, 10) * 5 + 3
        Xs, mean, std = standardize_features(X)
        np.testing.assert_allclose(Xs.mean(axis=0), np.zeros(10), atol=1e-5)

    def test_unit_std(self):
        X = np.random.randn(100, 10) * 5
        Xs, _, _ = standardize_features(X)
        np.testing.assert_allclose(Xs.std(axis=0), np.ones(10), atol=1e-4)

    def test_provided_stats(self):
        X_train = np.random.randn(80, 5)
        _, mean, std = standardize_features(X_train)
        X_test = np.random.randn(20, 5) * 3
        Xs, _, _ = standardize_features(X_test, mean=mean, std=std)
        # Not necessarily zero-mean, but shape should match
        assert Xs.shape == X_test.shape


# ---------------------------------------------------------------------------
# extract_tabular_features
# ---------------------------------------------------------------------------

class TestExtractTabularFeatures:
    def test_output_length(self):
        sig = np.random.randn(512)
        feats = extract_tabular_features(sig)
        assert feats.ndim == 1
        assert len(feats) == 12

    def test_finite_values(self):
        sig = np.random.randn(512)
        feats = extract_tabular_features(sig)
        assert np.all(np.isfinite(feats))


# ---------------------------------------------------------------------------
# augment_signal
# ---------------------------------------------------------------------------

class TestAugmentSignal:
    def test_output_shape(self):
        sig = np.random.randn(512).astype(np.float32)
        aug = augment_signal(sig)
        assert aug.shape == sig.shape

    def test_output_differs_from_input(self):
        sig = np.random.randn(512).astype(np.float32)
        rng = np.random.default_rng(0)
        aug = augment_signal(sig, rng=rng)
        # With non-zero noise and jitter, output should differ
        assert not np.array_equal(sig, aug)

    def test_deterministic_with_seed(self):
        sig = np.random.randn(128).astype(np.float32)
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        a1 = augment_signal(sig, rng=rng1)
        a2 = augment_signal(sig, rng=rng2)
        np.testing.assert_array_equal(a1, a2)


# ---------------------------------------------------------------------------
# batch_preprocess
# ---------------------------------------------------------------------------

class TestBatchPreprocess:
    def test_output_shape(self):
        signals = np.random.randn(8, 512).astype(np.float32)
        specs = batch_preprocess(signals, n_fft=64, hop_length=32)
        assert specs.ndim == 3
        assert specs.shape[0] == 8

    def test_augment_does_not_crash(self):
        signals = np.random.randn(4, 256).astype(np.float32)
        specs = batch_preprocess(signals, n_fft=64, hop_length=16, augment=True)
        assert specs.shape[0] == 4
