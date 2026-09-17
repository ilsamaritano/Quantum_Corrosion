"""Helper utilities for hybrid notebooks.
Provides a lightweight interface to build the model and run a quick forward pass.
This module reuses classes from `train_hybrid_9class.py` to avoid duplication.
"""
import torch
import numpy as np
from pathlib import Path

try:
    from train_hybrid_9class import HybridEfficientNetQNN_9Class
except Exception:
    # Fallback: define a minimal placeholder model if import fails
    import torch.nn as nn
    import torch.nn.functional as F
    class HybridEfficientNetQNN_9Class(nn.Module):
        def __init__(self, n_qubits=8, n_layers=2):
            super().__init__()
            self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(8, 9)
        def forward(self, x):
            x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
            x = x.repeat(1,3,1,1) if x.shape[1]==1 else x
            x = self.conv(x)
            x = self.pool(x).view(x.size(0), -1)
            return self.fc(x)


def build_model(device=None):
    """Build and return the hybrid model on the given device (or CPU).
    Returns: model (torch.nn.Module)
    """
    model = HybridEfficientNetQNN_9Class()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def random_spec_batch(batch_size=2):
    """Create a random spectrogram batch with shape (batch, 1, 16, 512).
    Matches the preprocessing used in the training scripts.
    """
    return torch.randn(batch_size, 1, 16, 512, dtype=torch.float32)


def smoke_forward(device=None):
    """Run a single forward pass to sanity-check model construction and shapes."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    x = random_spec_batch(2).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
    return out.cpu().numpy()


if __name__ == "__main__":
    import sys
    try:
        out = smoke_forward()
        print("Smoke forward output shape:", out.shape)
    except Exception as e:
        print("Smoke forward failed:", e)
