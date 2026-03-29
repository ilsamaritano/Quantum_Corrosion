"""Classical model baselines: ResNet-34, MobileNetV2, SimpleCNN, MLP."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch / torchvision not available — classical models disabled.")


# ---------------------------------------------------------------------------
# ResNet-34
# ---------------------------------------------------------------------------

def build_resnet34(
    num_classes: int,
    pretrained: bool = False,
    input_channels: int = 3,
) -> "nn.Module":
    """Build a ResNet-34 for multi-class classification.

    Args:
        num_classes: Number of output classes.
        pretrained: Load ImageNet pretrained weights (requires internet).
        input_channels: Number of input channels (3 for RGB spectrograms).

    Returns:
        A :class:`torch.nn.Module` with the final layer replaced.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for build_resnet34.")

    weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
    model = tv_models.resnet34(weights=weights)

    # Adapt first conv if not RGB
    if input_channels != 3:
        old = model.conv1
        model.conv1 = nn.Conv2d(
            input_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    logger.info(
        "Built ResNet-34: %d classes, pretrained=%s, channels=%d",
        num_classes,
        pretrained,
        input_channels,
    )
    return model


# ---------------------------------------------------------------------------
# MobileNetV2
# ---------------------------------------------------------------------------

def build_mobilenetv2(
    num_classes: int,
    pretrained: bool = False,
    input_channels: int = 3,
) -> "nn.Module":
    """Build a MobileNetV2 for multi-class classification.

    Args:
        num_classes: Number of output classes.
        pretrained: Load ImageNet pretrained weights.
        input_channels: Number of input channels.

    Returns:
        A :class:`torch.nn.Module` with the classifier head replaced.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for build_mobilenetv2.")

    weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = tv_models.mobilenet_v2(weights=weights)

    if input_channels != 3:
        old = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            input_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    logger.info(
        "Built MobileNetV2: %d classes, pretrained=%s, channels=%d",
        num_classes,
        pretrained,
        input_channels,
    )
    return model


# ---------------------------------------------------------------------------
# Simple CNN baseline
# ---------------------------------------------------------------------------

def build_simple_cnn(
    num_classes: int,
    input_channels: int = 1,
    input_size: int = 64,
) -> "nn.Module":
    """Build a lightweight CNN baseline for spectrogram classification.

    Architecture: ``Conv → BN → ReLU → MaxPool`` × 3 → ``Flatten → FC → FC``

    Args:
        num_classes: Number of output classes.
        input_channels: Number of input channels.
        input_size: Assumed square input spatial size (used only for logging).

    Returns:
        A :class:`torch.nn.Module`.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for build_simple_cnn.")

    class SimpleCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.classifier(self.features(x))

    model = SimpleCNN()
    logger.info(
        "Built SimpleCNN: %d classes, channels=%d", num_classes, input_channels
    )
    return model


# ---------------------------------------------------------------------------
# MLP classifier (tabular / reduced features)
# ---------------------------------------------------------------------------

def build_mlp_classifier(
    input_dim: int,
    num_classes: int,
    hidden_dims: List[int] = None,
) -> "nn.Module":
    """Build a fully-connected MLP for tabular or PCA-reduced features.

    Args:
        input_dim: Dimension of input features.
        num_classes: Number of output classes.
        hidden_dims: List of hidden layer sizes.  Defaults to ``[128, 64]``.

    Returns:
        A :class:`torch.nn.Module`.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for build_mlp_classifier.")

    if hidden_dims is None:
        hidden_dims = [128, 64]

    layers: List[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
        in_dim = h
    layers.append(nn.Linear(in_dim, num_classes))

    model = nn.Sequential(*layers)
    logger.info(
        "Built MLP: input_dim=%d, hidden=%s, classes=%d",
        input_dim,
        hidden_dims,
        num_classes,
    )
    return model


# ---------------------------------------------------------------------------
# Model introspection helpers
# ---------------------------------------------------------------------------

def count_trainable_params(model: "nn.Module") -> int:
    """Count the number of trainable parameters in a PyTorch model.

    Args:
        model: A :class:`torch.nn.Module`.

    Returns:
        Total count of parameters where ``requires_grad=True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(
    model: "nn.Module",
    input_size: tuple,
) -> Dict:
    """Produce a summary dict with layer names, output shapes and param counts.

    Args:
        model: A :class:`torch.nn.Module`.
        input_size: Input tensor size as a tuple, e.g. ``(1, 3, 64, 64)``.

    Returns:
        Dictionary with keys ``"layers"``, ``"total_params"``,
        ``"trainable_params"``.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for get_model_summary.")

    import torch

    summary: Dict = {"layers": [], "total_params": 0, "trainable_params": 0}

    hooks = []
    layer_info: List[Dict] = []

    def make_hook(name: str):
        def hook(module, inp, out):
            out_shape = list(out.shape) if isinstance(out, torch.Tensor) else "N/A"
            params = sum(p.numel() for p in module.parameters(recurse=False))
            layer_info.append(
                {"name": name, "type": type(module).__name__,
                 "output_shape": out_shape, "params": params}
            )
        return hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        x = torch.zeros(*input_size)
        model(x)
    except Exception as exc:
        logger.debug("get_model_summary forward pass error: %s", exc)
    finally:
        for h in hooks:
            h.remove()

    summary["layers"] = layer_info
    summary["total_params"] = sum(p.numel() for p in model.parameters())
    summary["trainable_params"] = count_trainable_params(model)
    return summary
