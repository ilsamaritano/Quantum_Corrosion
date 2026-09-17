"""
Quantum Corrosion Classification — Configuration
=================================================
All hyperparameters, paths, and experimental settings for the
hybrid quantum-classical radar-IQ corrosion classification pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
try:
    import torch
    _CUDA = torch.cuda.is_available()
except ImportError:
    _CUDA = False


@dataclass
class PathConfig:
    """File-system paths."""
    data_raw: Path = Path("data/raw")            # folders: 0.5g/ 1g/ 1.5g/ …
    data_processed: Path = Path("data/processed") # .npy spectrogram arrays
    data_splits: Path = Path("data/splits")        # train/val/test indices
    results: Path = Path("results")
    figures: Path = Path("results/figures")
    models: Path = Path("results/models")
    metrics: Path = Path("results")


@dataclass
class SignalConfig:
    """IQ-signal preprocessing."""
    sample_rate: float = 20e6          # 20 MHz USRP sampling
    fft_size: int = 4096               # per-frame FFT bins
    n_stacks: int = 128                # consecutive FFT frames per image
    center_freq: float = 5e9           # 2f0 = 5 GHz (harmonic)
    dtype: str = "complex64"           # IQ sample dtype in binary files
    window: str = "hann"               # FFT window function
    overlap_ratio: float = 0.25        # modest overlap to preserve transitions
    norm_method: str = "log"           # "minmax" | "zscore" | "log"
    img_size: Tuple[int, int] = (224, 224)  # resized spectrogram


@dataclass
class AugmentConfig:
    """Signal-level data augmentation."""
    enabled: bool = True
    time_shift_max: int = 512          # samples
    amplitude_scale: Tuple[float, float] = (0.9, 1.1)
    gaussian_noise_std: float = 0.01
    phase_jitter_std: float = 0.05     # radians
    prob: float = 0.8                  # Aumentato a 0.8 per regolarizzare l'enorme mole di dati


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    seed: int = 42
    n_rounds: int = 2                  # 2 rounds to ensure ~2 hours of training
    epochs: int = 150                  # 150 epochs are sufficient for this timeframe
    batch_size: int = 128              # Aumentato a 128 per massimizzare l'uso della VRAM dell'A6000 (48GB)
    lr: float = 3e-3                   # Leggermente più alto per il training ibrido veloce
    weight_decay: float = 1e-4
    scheduler: str = "cosine"          # "cosine" | "step" | "plateau"
    patience: int = 50                 # Maggior patience per evitare early stop prematuro nel lungo periodo
    num_workers: int = 16              # Aumentato a 16 per alimentare saturare le code della GPU
    pin_memory: bool = True
    mixed_precision: bool = True       # AMP for faster training
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1       # Aumentato per migliore regolarizzazione ed F1 score
    split_ratios: Tuple[float, float, float] = (0.60, 0.20, 0.20)
    device: str = "cuda" if _CUDA else "cpu"


@dataclass
class ClassicalModelConfig:
    """Classical CNN baselines."""
    architectures: List[str] = field(default_factory=lambda: [
        "resnet18", "resnet34", "resnet50", "resnet101",
        "mobilenetv2", "efficientnet_b0",
    ])
    pretrained: bool = True            # Inizializzato con pesi pre-addestrati ImageNet per estrazione feature migliore
    in_channels: int = 1               # grayscale spectrograms
    dropout: float = 0.3


@dataclass
class QuantumModelConfig:
    """Hybrid quantum-classical model."""
    n_qubits: int = 8
    n_layers: int = 6                  # slightly deeper circuit for more capacity
    encoding: str = "amplitude"       # Best tuned encoding
    entanglement: str = "linear"      # Best tuned entanglement
    measurement: str = "probs"         # "probs" | "expval"
    pca_dim: int = 1024                # Raddoppiato a 1024 per immagazzinare molte più peculiarità nello spazio limitato del VQC
    latent_dim: int = 2**8             # = 256
    classical_head: List[int] = field(default_factory=lambda: [1024, 512, 256])
    optimizer: str = "adam"            # "adam" | "spsa"
    lr: float = 1e-3                   # Best tuned learning rate
    shots: Optional[int] = None        # None = analytic, int = shot-based
    diff_method: str = "backprop"      # "backprop" accelera MASSIVAMENTE il calcolo dei gradienti PyTorch-native
    backend: str = "default.qubit"     # default.qubit gestisce backprop+amplitude su PyTorch
    redundancy_stride: int = 1         # 1: sfrutattamento dati 100% per saturare il lungo training
    use_class_weights: bool = True
    auto_tune: bool = False
    tune_encodings: List[str] = field(default_factory=lambda: ["amplitude", "angle", "iqp"])
    tune_layers: List[int] = field(default_factory=lambda: [4, 6, 8, 10])
    tune_entanglement: List[str] = field(default_factory=lambda: ["linear", "circular", "full"])
    tune_lrs: List[float] = field(default_factory=lambda: [1e-3, 3e-3])
    tuning_max_trials: int = 8
    tuning_epochs: int = 30
    ensemble_top_k: int = 3



@dataclass
class AblationConfig:
    """Ablation / data-efficiency experiments."""
    data_fractions: List[float] = field(default_factory=lambda: [
        0.10, 0.40, 1.0
    ])
    circuit_depths: List[int] = field(default_factory=lambda: [1, 2, 4, 6, 8])
    qubit_counts: List[int] = field(default_factory=lambda: [4, 6, 8, 10])
    encodings: List[str] = field(default_factory=lambda: [
        "amplitude", "angle", "iqp"
    ])
    n_repeats: int = 1                 # seeds per fraction (quick benchmark)


@dataclass
class PlotConfig:
    """Publication-quality figure settings."""
    format: str = "pdf"
    dpi: int = 300
    font_family: str = "serif"
    font_size: int = 14
    title_size: int = 16
    tick_size: int = 12
    legend_size: int = 12
    linewidth: float = 2.0
    figsize_single: Tuple[float, float] = (6, 4.5)
    figsize_double: Tuple[float, float] = (12, 4.5)
    colormap: str = "viridis"
    colors: List[str] = field(default_factory=lambda: [
        "#2196F3", "#F44336", "#4CAF50", "#FF9800",
        "#9C27B0", "#00BCD4", "#795548", "#607D8B",
    ])
    grid_alpha: float = 0.3


@dataclass
class Config:
    """Master configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    classical: ClassicalModelConfig = field(default_factory=ClassicalModelConfig)
    quantum: QuantumModelConfig = field(default_factory=QuantumModelConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)

    # Dataset class names derived from folder names
    class_names: List[str] = field(default_factory=lambda: [
        "0.5", "1", "1.5", "2", "2.5"
    ])

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def ensure_dirs(self):
        for p in [self.paths.data_processed, self.paths.data_splits,
                  self.paths.results, self.paths.figures, self.paths.models,
                  self.paths.metrics]:
            p.mkdir(parents=True, exist_ok=True)


def get_config(**overrides) -> Config:
    cfg = Config()
    cfg.ensure_dirs()
    # Apply any runtime overrides
    for k, v in overrides.items():
        parts = k.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return cfg
