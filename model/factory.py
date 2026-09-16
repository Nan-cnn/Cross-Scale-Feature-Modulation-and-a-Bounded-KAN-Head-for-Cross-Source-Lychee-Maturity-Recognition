"""Central model registry with an explicit no-pretraining contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import torch.nn as nn
from torchvision import models

from .MobileNetV2KANPlus import MobileNetV2KANPlus


FORBID_PRETRAINED_WEIGHTS = True


def _replace_classifier(model: nn.Module, architecture: str, num_classes: int) -> nn.Module:
    if architecture == "vgg16":
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif architecture in {"efficientnet_b0", "efficientnet_b3"}:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported classifier replacement: {architecture}")
    return model


def build_model(spec: Dict[str, Any], num_classes: int = 2) -> nn.Module:
    """Build one randomly initialized model from a serializable specification."""

    spec = deepcopy(spec)
    architecture = str(spec.pop("architecture"))
    spec.pop("display_name", None)
    if any(key in spec for key in ("weights", "pretrained", "checkpoint")):
        raise ValueError("Model specs may not request pretrained weights or checkpoints")

    if architecture == "mobilenetv2_kan_plus":
        return MobileNetV2KANPlus(num_classes=num_classes, **spec)
    if architecture == "mobilenet_v2":
        return models.mobilenet_v2(weights=None, num_classes=num_classes)
    if architecture == "resnet50":
        return models.resnet50(weights=None, num_classes=num_classes)
    if architecture == "vgg16":
        return _replace_classifier(models.vgg16(weights=None), architecture, num_classes)
    if architecture == "efficientnet_b0":
        return _replace_classifier(models.efficientnet_b0(weights=None), architecture, num_classes)
    if architecture == "efficientnet_b3":
        return _replace_classifier(models.efficientnet_b3(weights=None), architecture, num_classes)
    if architecture == "shufflenet_v2_x1_0":
        return models.shufflenet_v2_x1_0(weights=None, num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {architecture}")


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_size_megabytes(model: nn.Module) -> float:
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (parameter_bytes + buffer_bytes) / (1024.0 ** 2)


__all__ = [
    "FORBID_PRETRAINED_WEIGHTS",
    "build_model",
    "model_size_megabytes",
    "trainable_parameter_count",
]
