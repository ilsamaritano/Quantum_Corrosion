"""
Classical Models — CNN baselines for spectrogram classification
===============================================================
ResNet-{18,34,50,101}, MobileNetV2, EfficientNet-B0.
All adapted for single-channel grayscale input.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _adapt_first_conv(model: nn.Module, in_channels: int = 1) -> nn.Module:
    """
    Replace the first convolutional layer to accept grayscale input.
    Preserves kernel size, stride, padding — just changes in_channels.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            old = module
            new_conv = nn.Conv2d(
                in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=(old.bias is not None),
            )
            # Initialize by averaging pretrained RGB weights
            if old.in_channels == 3 and in_channels == 1:
                with torch.no_grad():
                    new_conv.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            # Set attribute on parent
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_conv)
            break
    return model


def _replace_classifier(model: nn.Module, n_classes: int, dropout: float = 0.3) -> nn.Module:
    """Replace the final FC layer(s) for our number of classes."""
    # ResNet family
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, n_classes),
        )
    # MobileNetV2
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, n_classes)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, n_classes),
            )
    return model


# ── Model builders ──────────────────────────────────────────────────────────

def build_resnet(
    variant: int = 34,
    n_classes: int = 5,
    in_channels: int = 1,
    pretrained: bool = False,
    dropout: float = 0.3,
) -> nn.Module:
    factory = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
    }
    weights = "IMAGENET1K_V1" if pretrained else None
    model = factory[variant](weights=weights)
    model = _adapt_first_conv(model, in_channels)
    model = _replace_classifier(model, n_classes, dropout)
    return model


def build_mobilenetv2(
    n_classes: int = 5,
    in_channels: int = 1,
    pretrained: bool = False,
    dropout: float = 0.3,
) -> nn.Module:
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model = _adapt_first_conv(model, in_channels)
    model = _replace_classifier(model, n_classes, dropout)
    return model


def build_efficientnet_b0(
    n_classes: int = 5,
    in_channels: int = 1,
    pretrained: bool = False,
    dropout: float = 0.3,
) -> nn.Module:
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model = _adapt_first_conv(model, in_channels)
    # EfficientNet classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, n_classes),
    )
    return model


# ── Unified factory ─────────────────────────────────────────────────────────

def build_classical_model(
    arch: str,
    n_classes: int = 5,
    in_channels: int = 1,
    pretrained: bool = False,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Unified factory: 'resnet18', 'resnet34', 'resnet50', 'resnet101',
                     'mobilenetv2', 'efficientnet_b0'.
    """
    arch = arch.lower().replace("-", "").replace("_", "")
    if arch.startswith("resnet"):
        variant = int(arch.replace("resnet", ""))
        return build_resnet(variant, n_classes, in_channels, pretrained, dropout)
    elif arch == "mobilenetv2":
        return build_mobilenetv2(n_classes, in_channels, pretrained, dropout)
    elif arch in ("efficientnetb0", "efficientnet_b0"):
        return build_efficientnet_b0(n_classes, in_channels, pretrained, dropout)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ── Utilities ───────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def model_summary(model: nn.Module, name: str = "") -> str:
    """Quick text summary."""
    params = count_parameters(model)
    lines = [
        f"{'='*60}",
        f"Model: {name or model.__class__.__name__}",
        f"  Total params:     {params['total']:>12,}",
        f"  Trainable params: {params['trainable']:>12,}",
        f"  Frozen params:    {params['frozen']:>12,}",
        f"{'='*60}",
    ]
    return "\n".join(lines)
