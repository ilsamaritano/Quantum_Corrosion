"""Signal preprocessing: FFT, STFT spectrograms, feature extraction, augmentation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def compute_fft(
    signal: np.ndarray,
    n_fft: Optional[int] = None,
    return_complex: bool = False,
) -> np.ndarray:
    """Compute the FFT of an IQ signal.

    Args:
        signal: 1-D real or complex array.
        n_fft: FFT size.  Defaults to the signal length.
        return_complex: If ``True``, return the complex spectrum; otherwise
            return the magnitude.

    Returns:
        Magnitude spectrum (or complex spectrum) of length ``n_fft // 2 + 1``
        for real signals, or ``n_fft`` for complex signals.
    """
    signal = np.asarray(signal)
    n_fft = n_fft or len(signal)

    spectrum = np.fft.fft(signal, n=n_fft)
    if return_complex:
        return spectrum

    # Return single-sided magnitude for real signals
    mag = np.abs(spectrum[: n_fft // 2 + 1])
    return mag


def generate_spectrogram(
    signal: np.ndarray,
    n_fft: int = 256,
    hop_length: int = 64,
    n_mels: Optional[int] = None,
    sr: int = 1000,
    normalize: bool = True,
) -> np.ndarray:
    """Generate an STFT spectrogram from an IQ signal.

    Args:
        signal: 1-D float array (real-valued; imaginary part is handled via
            the magnitude of the analytic signal if present).
        n_fft: FFT window size.
        hop_length: Number of samples between successive frames.
        n_mels: If given, project to a mel filterbank of this size.
        sr: Sample rate in Hz (used for mel filterbank).
        normalize: If ``True``, convert to dB and normalise to ``[0, 1]``.

    Returns:
        2-D array of shape ``(freq_bins, time_frames)``.
    """
    from .spectrograms import stft_spectrogram, mel_spectrogram  # avoid circular

    if n_mels is not None:
        spec = mel_spectrogram(
            signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
    else:
        spec = stft_spectrogram(
            signal, n_fft=n_fft, hop_length=hop_length, normalize=normalize
        )
    return spec


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_spectrogram_images(
    spectrograms: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path,
    class_names: Optional[List[str]] = None,
    fmt: str = "png",
) -> None:
    """Save spectrogram arrays as image files organised by class.

    Args:
        spectrograms: Array of shape ``(N, freq, time)`` or ``(N, H, W)``.
        labels: Integer label array of shape ``(N,)``.
        output_dir: Root directory; sub-directories per class are created.
        class_names: Optional list mapping label integer → class name string.
        fmt: Image format, e.g. ``"png"`` or ``"jpg"``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    unique_labels = np.unique(labels)

    for lbl in unique_labels:
        name = class_names[lbl] if class_names and lbl < len(class_names) else str(lbl)
        cls_dir = output_dir / name
        cls_dir.mkdir(parents=True, exist_ok=True)

        idxs = np.where(labels == lbl)[0]
        for j, idx in enumerate(idxs):
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(spectrograms[idx], aspect="auto", origin="lower", cmap="viridis")
            ax.axis("off")
            fig.savefig(cls_dir / f"{j:04d}.{fmt}", bbox_inches="tight", pad_inches=0)
            plt.close(fig)

    logger.info("Saved %d spectrogram images to %s", len(spectrograms), output_dir)


# ---------------------------------------------------------------------------
# Feature standardisation
# ---------------------------------------------------------------------------

def standardize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardise a feature matrix.

    Args:
        features: 2-D array ``(n_samples, n_features)``.
        mean: Pre-computed mean.  If ``None``, computed from *features*.
        std: Pre-computed standard deviation.  If ``None``, computed from
            *features*.

    Returns:
        Tuple ``(standardized, mean, std)``.
    """
    features = np.asarray(features, dtype=np.float64)
    if mean is None:
        mean = features.mean(axis=0)
    if std is None:
        std = features.std(axis=0)

    std_safe = np.where(std == 0, 1.0, std)
    return (features - mean) / std_safe, mean, std


# ---------------------------------------------------------------------------
# Tabular feature extraction
# ---------------------------------------------------------------------------

def extract_tabular_features(signal: np.ndarray) -> np.ndarray:
    """Extract a 1-D vector of hand-crafted statistical and spectral features.

    Features (in order):
    - mean, std, skewness, kurtosis, peak-to-peak, RMS
    - spectral centroid, spectral bandwidth, spectral roll-off (85 %)
    - top-3 spectral peak magnitudes

    Args:
        signal: 1-D float array.

    Returns:
        1-D feature vector of length 15.
    """
    from scipy.stats import kurtosis as _kurt, skew as _skew

    s = np.asarray(signal, dtype=np.float64).ravel()
    n = len(s)

    # Time-domain statistics
    mean = float(s.mean())
    std = float(s.std())
    skewness = float(_skew(s))
    kurt = float(_kurt(s))
    ptp = float(s.max() - s.min())
    rms = float(np.sqrt(np.mean(s**2)))

    # Spectral features
    mag = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(n)
    total_power = mag.sum() + 1e-12
    centroid = float(np.dot(freqs, mag) / total_power)
    bandwidth = float(
        np.sqrt(np.dot((freqs - centroid) ** 2, mag) / total_power)
    )
    cum_power = np.cumsum(mag)
    rolloff_idx = np.searchsorted(cum_power, 0.85 * cum_power[-1])
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # Top-3 spectral peak magnitudes
    top3 = np.sort(mag)[-3:][::-1]

    return np.array(
        [mean, std, skewness, kurt, ptp, rms,
         centroid, bandwidth, rolloff,
         float(top3[0]), float(top3[1]), float(top3[2])],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def augment_signal(
    signal: np.ndarray,
    jitter_std: float = 0.01,
    shift_max: int = 10,
    noise_std: float = 0.005,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply a random composition of augmentations to an IQ signal.

    Augmentations applied (all random):
    - Amplitude jitter (additive Gaussian noise proportional to signal std)
    - Circular time shift
    - Additive white Gaussian noise
    - Random amplitude scaling

    Args:
        signal: 1-D float array.
        jitter_std: Standard deviation of the jitter noise relative to signal
            std.
        shift_max: Maximum circular shift in samples.
        noise_std: Standard deviation of additive white noise.
        scale_range: ``(lo, hi)`` range for amplitude scaling factor.
        rng: Optional :class:`numpy.random.Generator` for reproducibility.

    Returns:
        Augmented signal as a float32 array.
    """
    if rng is None:
        rng = np.random.default_rng()

    s = np.asarray(signal, dtype=np.float64).copy()

    # Amplitude jitter
    s += rng.normal(0, jitter_std * (s.std() + 1e-8), s.shape)

    # Circular time shift
    shift = int(rng.integers(-shift_max, shift_max + 1))
    s = np.roll(s, shift)

    # Additive white noise
    s += rng.normal(0, noise_std, s.shape)

    # Amplitude scaling
    scale = rng.uniform(scale_range[0], scale_range[1])
    s *= scale

    return s.astype(np.float32)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def batch_preprocess(
    signals: np.ndarray,
    n_fft: int = 256,
    hop_length: int = 64,
    augment: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """Preprocess a batch of signals to STFT spectrograms.

    Args:
        signals: 2-D array ``(n_samples, signal_length)``.
        n_fft: FFT window size.
        hop_length: STFT hop length.
        augment: If ``True``, apply random augmentation before computing the
            spectrogram.
        normalize: Normalise spectrogram to ``[0, 1]``.

    Returns:
        3-D array ``(n_samples, freq_bins, time_frames)``.
    """
    from .spectrograms import stft_spectrogram

    specs = []
    rng = np.random.default_rng()
    for sig in signals:
        if augment:
            sig = augment_signal(sig, rng=rng)
        spec = stft_spectrogram(sig, n_fft=n_fft, hop_length=hop_length,
                                normalize=normalize)
        specs.append(spec)
    return np.stack(specs)
