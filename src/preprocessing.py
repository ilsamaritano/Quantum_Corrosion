"""
Preprocessing — Memory-efficient IQ → Spectrogram pipeline
===========================================================
Handles multi-GB binary IQ files via memory-mapped streaming.
Produces normalised grayscale spectrogram images ready for CNN/VQC.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Generator
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from tqdm import tqdm
from scipy.signal import get_window
from PIL import Image

from src.config import Config, SignalConfig, AugmentConfig

logger = logging.getLogger(__name__)


# ── IQ Loading (memory-mapped for 3 GB+ files) ─────────────────────────────

def load_iq_mmap(filepath: Path, dtype: str = "complex64") -> np.memmap:
    """Memory-map a binary IQ file — zero RAM overhead on open."""
    np_dtype = np.dtype(dtype)
    file_size = filepath.stat().st_size
    n_samples = file_size // np_dtype.itemsize
    return np.memmap(filepath, dtype=np_dtype, mode='r', shape=(n_samples,))


def iq_frame_generator(
    iq: np.memmap,
    fft_size: int,
    n_stacks: int,
    overlap_ratio: float = 0.0,
) -> Generator[np.ndarray, None, None]:
    """
    Yield (n_stacks, fft_size) blocks from a memory-mapped IQ stream.
    Each block becomes one spectrogram image.
    """
    hop = int(fft_size * (1.0 - overlap_ratio))
    samples_per_image = n_stacks * hop + (fft_size - hop)  # account for last frame
    total = len(iq)
    start = 0
    while start + samples_per_image <= total:
        frames = np.empty((n_stacks, fft_size), dtype=iq.dtype)
        for i in range(n_stacks):
            s = start + i * hop
            frames[i] = iq[s : s + fft_size]
        yield frames
        start += samples_per_image


# ── FFT & Spectrogram ───────────────────────────────────────────────────────

def compute_spectrogram(
    frames: np.ndarray,
    window: str = "hann",
    norm_method: str = "minmax",
) -> np.ndarray:
    """
    Convert (n_stacks, fft_size) IQ frames → 3-channel rich representation.
    Returns:
        3-D float32 array in [0, 1] range, shape (3, n_stacks, fft_size) 
        containing [Magnitude, Phase, Instantaneous Frequency/Derivative].
    """
    win = get_window(window, frames.shape[1]).astype(np.float32)
    windowed = frames * win[np.newaxis, :]
    spectrum = np.fft.fftshift(np.fft.fft(windowed, axis=1), axes=1)
    
    # Channel 0: Magnitude (Log norm)
    magnitude = np.abs(spectrum).astype(np.float32)
    magnitude = 20.0 * np.log10(magnitude + 1e-12)
    vmin = magnitude.max() - 80.0
    magnitude = np.clip(magnitude, vmin, None)
    mn, mx = magnitude.min(), magnitude.max()
    if mx - mn > 1e-10:
        magnitude = (magnitude - mn) / (mx - mn)
    else:
        magnitude = np.zeros_like(magnitude)
        
    # Channel 1: Phase (Norm to 0-1)
    phase = np.angle(spectrum).astype(np.float32)
    phase = (phase + np.pi) / (2 * np.pi)
    
    # Channel 2: Phase Derivative (Temporal changes in phase, scaled)
    phase_diff = np.diff(phase, axis=0, prepend=phase[0:1, :])
    phase_diff = (phase_diff + 1.0) / 2.0 # diff is in [-1, 1], scale to [0, 1]
    
    # Stack into (3, H, W)
    rich_spec = np.stack([magnitude, phase, phase_diff], axis=0)
    return rich_spec

def spectrogram_to_image(
    spec: np.ndarray,
    size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Resize 3-channel spectrogram to target image size, return uint8."""
    # spec is (3, H, W)
    out = np.zeros((3, size[1], size[0]), dtype=np.uint8)
    for c in range(3):
        img_c = Image.fromarray((spec[c] * 255).astype(np.uint8), mode='L')
        img_c = img_c.resize(size, Image.LANCZOS)
        out[c] = np.array(img_c)
    return out


# ── Single-file processing ──────────────────────────────────────────────────

def process_single_file(
    filepath: Path,
    label: int,
    cfg: SignalConfig,
    augment_cfg: Optional[AugmentConfig],
    output_dir: Path,
    file_idx: int = 0,
) -> List[dict]:
    """
    Process one .iq file → multiple spectrogram .npy files.
    Returns list of {path, label} metadata dicts.
    """
    logger.info(f"Processing {filepath.name} (label={label})")
    iq = load_iq_mmap(filepath, cfg.dtype)
    records = []
    gen = iq_frame_generator(iq, cfg.fft_size, cfg.n_stacks, cfg.overlap_ratio)

    for img_idx, frames in enumerate(gen):
        if augment_cfg is not None and augment_cfg.enabled:
            frames = augment_iq_signal(frames, augment_cfg)
        spec = compute_spectrogram(frames, cfg.window, cfg.norm_method)
        img = spectrogram_to_image(spec, cfg.img_size)

        fname = f"label{label}_file{file_idx}_img{img_idx:06d}.npy"
        out_path = output_dir / fname
        np.save(out_path, img)
        records.append({"path": str(out_path), "label": label})

    del iq  # release mmap
    return records


# ── Full dataset preprocessing (parallel) ───────────────────────────────────

def discover_iq_files(data_dir: Path, class_names: List[str]) -> List[Tuple[Path, int]]:
    """
    Scan data_dir for class folders and IQ binary files.
    Supports folder names like '0.5g', '1g', '1.5g', etc.
    Also tries numeric-only names like '0.5', '1', '1.5'.
    """
    files = []
    for label, cls_name in enumerate(class_names):
        # Try multiple folder naming conventions
        candidates = [
            data_dir / cls_name,
            data_dir / cls_name.replace("g", ""),
            data_dir / cls_name.replace(".", "_"),
            data_dir / cls_name.replace(".0g", "g"),
        ]
        cls_dir = None
        for c in candidates:
            if c.exists():
                cls_dir = c
                break
        if cls_dir is None:
            logger.warning(f"Class folder not found for '{cls_name}', tried: {candidates}")
            continue

        iq_files = sorted(
            [f for f in cls_dir.iterdir()
             if f.is_file() and f.suffix in ('.iq', '.raw', '.bin', '')]
        )
        for f in iq_files:
            # Skip tiny files (< 1 MB likely metadata)
            if f.stat().st_size > 1_000_000:
                files.append((f, label))
        logger.info(f"  {cls_name}: found {len(iq_files)} IQ files in {cls_dir}")
    return files


def preprocess_dataset(cfg: Config, max_workers: int = 4) -> List[dict]:
    """
    Full preprocessing pipeline: discover files → parallel FFT → save spectrograms.
    Returns metadata list for dataset construction.
    """
    cfg.paths.data_processed.mkdir(parents=True, exist_ok=True)
    files = discover_iq_files(cfg.paths.data_raw, cfg.class_names)
    logger.info(f"Discovered {len(files)} IQ files across {cfg.n_classes} classes.")

    if not files:
        raise FileNotFoundError(
            f"No IQ files found in {cfg.paths.data_raw}. "
            f"Expected folders: {cfg.class_names}"
        )

    all_records = []

    # Parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, (fpath, label) in enumerate(files):
            fut = executor.submit(
                process_single_file,
                fpath, label, cfg.signal, cfg.augment,
                cfg.paths.data_processed, idx,
            )
            futures[fut] = (fpath, label)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            try:
                records = fut.result()
                all_records.extend(records)
            except Exception as e:
                fpath, label = futures[fut]
                logger.error(f"Failed on {fpath}: {e}")

    # Save metadata
    import json
    meta_path = cfg.paths.data_processed / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(all_records, f)
    logger.info(f"Preprocessed {len(all_records)} spectrograms → {cfg.paths.data_processed}")

    # Print class distribution
    from collections import Counter
    dist = Counter(r["label"] for r in all_records)
    for label, count in sorted(dist.items()):
        logger.info(f"  Class {cfg.class_names[label]}: {count} images")

    return all_records


# ── Signal-level augmentation ───────────────────────────────────────────────

def augment_iq_signal(
    frames: np.ndarray,
    cfg,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply signal-domain augmentation to IQ frames."""
    if rng is None:
        rng = np.random.default_rng()
    out = frames.copy()

    if rng.random() < cfg.prob:
        # Time shift
        shift = rng.integers(-cfg.time_shift_max, cfg.time_shift_max)
        out = np.roll(out, shift, axis=1)

    if rng.random() < cfg.prob:
        # Amplitude scaling
        scale = rng.uniform(*cfg.amplitude_scale)
        out = out * scale

    if rng.random() < cfg.prob:
        # Additive Gaussian noise
        noise = rng.normal(0, cfg.gaussian_noise_std, out.shape).astype(out.dtype)
        out = out + noise

    if rng.random() < cfg.prob:
        # Phase jitter
        phase = rng.normal(0, cfg.phase_jitter_std, out.shape).astype(np.float32)
        out = out * np.exp(1j * phase)

    return out


# ── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.config import get_config
    cfg = get_config()
    records = preprocess_dataset(cfg, max_workers=4)
    print(f"Total spectrograms: {len(records)}")
