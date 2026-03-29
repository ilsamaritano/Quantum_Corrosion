"""Dimensionality reduction: PCA and a simple MLP autoencoder."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------------

def fit_pca(
    features: np.ndarray,
    n_components: int = 16,
    whiten: bool = False,
) -> Tuple["sklearn.decomposition.PCA", np.ndarray]:  # type: ignore[name-defined]
    """Fit a PCA model on *features*.

    Args:
        features: 2-D array ``(n_samples, n_features)``.
        n_components: Number of principal components to keep.
        whiten: Whether to whiten the components.

    Returns:
        Tuple ``(pca_model, explained_variance_ratio)``.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, whiten=whiten)
    pca.fit(features)
    evr = pca.explained_variance_ratio_
    logger.info(
        "PCA fitted: %d components, cumulative explained variance = %.3f",
        n_components,
        evr.sum(),
    )
    return pca, evr


def transform_pca(
    features: np.ndarray,
    pca_model: "sklearn.decomposition.PCA",  # type: ignore[name-defined]
) -> np.ndarray:
    """Project *features* into PCA space.

    Args:
        features: 2-D float array ``(n_samples, n_features)``.
        pca_model: A fitted :class:`sklearn.decomposition.PCA` instance.

    Returns:
        Projected array ``(n_samples, n_components)``.
    """
    return pca_model.transform(features)


def fit_transform_pca(
    features: np.ndarray,
    n_components: int = 16,
) -> Tuple[np.ndarray, "sklearn.decomposition.PCA"]:  # type: ignore[name-defined]
    """Fit PCA and return projected features in one call.

    Args:
        features: 2-D float array ``(n_samples, n_features)``.
        n_components: Number of components.

    Returns:
        Tuple ``(projected_features, pca_model)``.
    """
    pca, _ = fit_pca(features, n_components=n_components)
    return transform_pca(features, pca), pca


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    class SimpleAutoencoder(nn.Module):
        """Symmetric MLP autoencoder for tabular feature compression.

        Architecture: ``input_dim → 128 → 64 → latent_dim → 64 → 128 → input_dim``

        Args:
            input_dim: Dimension of the input features.
            latent_dim: Bottleneck dimension.
        """

        def __init__(self, input_dim: int, latent_dim: int = 16) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, input_dim),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            return self.decoder(self.encoder(x))

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            return self.encoder(x)

    _TORCH_AVAILABLE = True

except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — autoencoder functionality disabled.")

    class SimpleAutoencoder:  # type: ignore[no-redef]
        """Stub when PyTorch is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for SimpleAutoencoder.")


def train_autoencoder(
    features: np.ndarray,
    latent_dim: int = 16,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
) -> Tuple["SimpleAutoencoder", List[float]]:
    """Train a :class:`SimpleAutoencoder` on *features*.

    Args:
        features: 2-D float array ``(n_samples, n_features)``.
        latent_dim: Bottleneck dimension.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Mini-batch size.
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        Tuple ``(trained_model, training_losses)``.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for train_autoencoder.")

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.tensor(features, dtype=torch.float32).to(device)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleAutoencoder(features.shape[1], latent_dim=latent_dim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    losses: List[float] = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimiser.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(batch)
        avg_loss = epoch_loss / len(X)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            logger.info("Autoencoder epoch %d/%d — loss: %.6f", epoch + 1, epochs, avg_loss)

    return model, losses


def encode_latent_space(
    autoencoder: "SimpleAutoencoder",
    features: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Encode *features* into the autoencoder latent space.

    Args:
        autoencoder: Trained :class:`SimpleAutoencoder`.
        features: 2-D float array ``(n_samples, n_features)``.
        device: Torch device string.

    Returns:
        Latent representation array ``(n_samples, latent_dim)``.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for encode_latent_space.")

    import torch

    autoencoder.eval()
    X = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        z = autoencoder.encode(X)
    return z.cpu().numpy()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_reducer(reducer: object, path: str | Path) -> None:
    """Persist a PCA or autoencoder model to *path* using pickle/torch.

    Args:
        reducer: PCA instance or :class:`SimpleAutoencoder` module.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if _TORCH_AVAILABLE:
        import torch

        if isinstance(reducer, torch.nn.Module):
            torch.save(reducer, path)
            logger.info("Saved autoencoder to %s", path)
            return

    with open(path, "wb") as fh:
        pickle.dump(reducer, fh)
    logger.info("Saved reducer to %s", path)


def load_reducer(path: str | Path) -> object:
    """Load a PCA or autoencoder model from *path*.

    Args:
        path: Source file path.

    Returns:
        Loaded reducer object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reducer file not found: {path}")

    if _TORCH_AVAILABLE:
        import torch

        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            pass

    with open(path, "rb") as fh:
        return pickle.load(fh)
