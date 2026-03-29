"""STFT/mel spectrogram computation, resizing, tensor conversion and plotting."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spectrogram computation
# ---------------------------------------------------------------------------

def stft_spectrogram(
    signal: np.ndarray,
    n_fft: int = 256,
    hop_length: int = 64,
    window: str = "hann",
    normalize: bool = True,
) -> np.ndarray:
    """Compute a power STFT spectrogram from a 1-D signal.

    Args:
        signal: 1-D float array.
        n_fft: FFT window size.
        hop_length: Samples between successive frames.
        window: Window function name — ``"hann"``, ``"hamming"``,
            ``"blackman"``, or ``"rectangular"``.
        normalize: If ``True``, convert to dB scale and map to ``[0, 1]``.

    Returns:
        2-D float32 array of shape ``(n_fft // 2 + 1, n_frames)``.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = len(signal)

    if window == "hann":
        win = np.hanning(n_fft)
    elif window == "hamming":
        win = np.hamming(n_fft)
    elif window == "blackman":
        win = np.blackman(n_fft)
    else:
        win = np.ones(n_fft)

    # Pad signal so every frame is complete
    n_frames = 1 + (n - n_fft) // hop_length
    if n_frames <= 0:
        pad_len = n_fft - n
        signal = np.pad(signal, (0, pad_len))
        n_frames = 1

    frames = np.stack(
        [
            signal[i * hop_length: i * hop_length + n_fft] * win
            for i in range(n_frames)
        ],
        axis=1,
    )  # shape (n_fft, n_frames)

    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=0)) ** 2  # power spectrum
    # shape (n_fft // 2 + 1, n_frames)

    if normalize:
        # Convert to dB (add small epsilon to avoid log(0))
        spec = 10 * np.log10(spec + 1e-10)
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)

    return spec.astype(np.float32)


def mel_spectrogram(
    signal: np.ndarray,
    sr: int = 1000,
    n_fft: int = 256,
    hop_length: int = 64,
    n_mels: int = 64,
) -> np.ndarray:
    """Compute a mel-frequency power spectrogram.

    A simple triangular mel filterbank is applied to the STFT power spectrum.

    Args:
        signal: 1-D float array.
        sr: Sample rate in Hz.
        n_fft: FFT window size.
        hop_length: Hop length.
        n_mels: Number of mel filter bands.

    Returns:
        2-D float32 array of shape ``(n_mels, n_frames)``.
    """
    power_spec = stft_spectrogram(
        signal, n_fft=n_fft, hop_length=hop_length, normalize=False
    )  # (freq_bins, n_frames)

    freq_bins = power_spec.shape[0]
    fmin, fmax = 0.0, sr / 2.0

    # Build mel filterbank (triangular filters)
    def hz_to_mel(f: float) -> float:
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m: float) -> float:
        return 700 * (10 ** (m / 2595) - 1)

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, freq_bins - 1)

    filterbank = np.zeros((n_mels, freq_bins))
    for m in range(1, n_mels + 1):
        f_start, f_mid, f_end = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if f_mid > f_start:
            filterbank[m - 1, f_start:f_mid] = (
                np.arange(f_start, f_mid) - f_start
            ) / (f_mid - f_start + 1e-8)
        if f_end > f_mid:
            filterbank[m - 1, f_mid:f_end] = (
                f_end - np.arange(f_mid, f_end)
            ) / (f_end - f_mid + 1e-8)

    mel_spec = filterbank @ power_spec  # (n_mels, n_frames)
    mel_spec = np.maximum(mel_spec, 1e-10)
    mel_spec = 10 * np.log10(mel_spec)
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
    return mel_spec.astype(np.float32)


# ---------------------------------------------------------------------------
# Resize / tensor conversion
# ---------------------------------------------------------------------------

def resize_spectrogram(
    spec: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Resize a spectrogram to *target_size* using bilinear interpolation.

    Args:
        spec: 2-D float array ``(freq_bins, time_frames)``.
        target_size: ``(height, width)`` target size.

    Returns:
        Resized 2-D float32 array.
    """
    from scipy.ndimage import zoom

    h_in, w_in = spec.shape
    h_out, w_out = target_size
    zoom_factors = (h_out / h_in, w_out / w_in)
    return zoom(spec.astype(np.float32), zoom_factors, order=1)


def spectrogram_to_tensor(
    spec: np.ndarray,
    normalize: bool = True,
) -> "torch.Tensor":
    """Convert a 2-D spectrogram array to a 3-channel PyTorch tensor.

    Replicates the single channel across 3 channels for compatibility with
    CNNs expecting RGB input.

    Args:
        spec: 2-D float array.
        normalize: Apply per-tensor min-max normalisation to ``[0, 1]``.

    Returns:
        Float tensor of shape ``(3, H, W)``.
    """
    import torch

    arr = np.asarray(spec, dtype=np.float32)
    if normalize:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
    return tensor


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_spectrogram(
    spec: np.ndarray,
    title: str = "",
    ax: Optional["matplotlib.axes.Axes"] = None,  # type: ignore[name-defined]
    colorbar: bool = True,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Plot a single spectrogram using Matplotlib.

    Args:
        spec: 2-D float array.
        title: Axes title.
        ax: Existing :class:`matplotlib.axes.Axes` to draw on.  A new figure
            is created when ``None``.
        colorbar: If ``True``, add a colour bar.

    Returns:
        The :class:`matplotlib.figure.Figure` containing the plot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    im = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Frequency bins")

    if colorbar:
        fig.colorbar(im, ax=ax)  # type: ignore[union-attr]

    return fig  # type: ignore[return-value]


def compare_spectrograms(
    signals_dict: dict,
    labels: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    n_examples: int = 3,
    n_fft: int = 256,
    hop_length: int = 64,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Plot raw IQ signals and corresponding spectrograms for multiple classes.

    Args:
        signals_dict: ``{class_name: array(n_samples, length)}``.
        labels: Unused (kept for API compatibility).
        class_names: Override display names for classes.
        n_examples: Number of example signals per class.
        n_fft: FFT window size for STFT.
        hop_length: STFT hop length.

    Returns:
        Matplotlib figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cls_names = class_names or list(signals_dict.keys())
    n_cls = len(cls_names)
    # Each class: 1 row for signal + 1 row for spectrogram
    fig, axes = plt.subplots(
        2 * n_cls,
        n_examples,
        figsize=(n_examples * 4, n_cls * 4),
    )
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    for ci, cname in enumerate(cls_names):
        arr = signals_dict.get(cname)
        if arr is None:
            continue
        for j in range(min(n_examples, len(arr))):
            sig = arr[j]
            # Signal row
            ax_sig = axes[2 * ci, j]
            ax_sig.plot(sig[:512])
            ax_sig.set_title(f"{cname} (signal {j})", fontsize=8)
            ax_sig.set_xlabel("Samples")

            # Spectrogram row
            spec = stft_spectrogram(sig, n_fft=n_fft, hop_length=hop_length)
            ax_spec = axes[2 * ci + 1, j]
            ax_spec.imshow(spec, aspect="auto", origin="lower", cmap="viridis")
            ax_spec.set_title(f"{cname} (spec {j})", fontsize=8)

    fig.tight_layout()
    return fig
