"""Enhanced MobileNetV2-KAN model for reproducible experiments.

The legacy project model only replaces the final classifier. This module keeps
MobileNetV2's efficient inverted residual backbone, adds lightweight
cross-scale feature modulation at three semantic stages, and explicitly keeps
KAN inputs inside the B-spline knot range. All weights start from scratch.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

from .KANLinear import KANLinear


class CrossScaleGatedFeatureModulation(nn.Module):
    """Fuse local and context depthwise features through a residual gate."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = self.fuse(torch.cat((self.local(x), self.context(x)), dim=1))
        return x + self.residual_scale * multi_scale * self.gate(multi_scale)


class MixStyle(nn.Module):
    """Randomize feature statistics during training without target-domain data."""

    def __init__(self, probability: float = 0.5, alpha: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.probability = float(probability)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or x.size(0) < 2 or torch.rand((), device=x.device) > self.probability:
            return x
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = (x.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps).sqrt()
        normalized = (x - mean) / std
        permutation = torch.randperm(x.size(0), device=x.device)
        # CUDA's Dirichlet sampler does not accept fp16 concentration tensors.
        alpha = torch.full(
            (x.size(0), 1, 1, 1), self.alpha, device=x.device, dtype=torch.float32
        )
        mixing = torch.distributions.Beta(alpha, alpha).sample().to(dtype=x.dtype)
        mixed_mean = mixing * mean + (1.0 - mixing) * mean[permutation]
        mixed_std = mixing * std + (1.0 - mixing) * std[permutation]
        return normalized * mixed_std + mixed_mean


class BoundedKANClassifier(nn.Module):
    """Configurable KAN head with inputs mapped to the declared knot range."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        bottleneck_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        head_type: str = "kan",
        kan_layers: int = 2,
        grid_size: int = 5,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        if head_type not in {"kan", "linear"}:
            raise ValueError("head_type must be 'kan' or 'linear'")
        if kan_layers not in {1, 2, 3}:
            raise ValueError("kan_layers must be one of 1, 2, or 3")
        if grid_size < 2 or spline_order < 1:
            raise ValueError("grid_size must be >= 2 and spline_order must be >= 1")

        self.head_type = head_type
        self.projection = nn.Sequential(
            nn.Linear(in_features, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(float(dropout))

        if head_type == "linear":
            self.linear = nn.Linear(bottleneck_dim, num_classes)
            self.kan = nn.ModuleList()
            self.intermediate_norms = nn.ModuleList()
            return

        if kan_layers == 1:
            dimensions = [bottleneck_dim, num_classes]
        elif kan_layers == 2:
            dimensions = [bottleneck_dim, hidden_dim, num_classes]
        else:
            dimensions = [bottleneck_dim, hidden_dim, max(32, hidden_dim // 2), num_classes]

        self.kan = nn.ModuleList(
            KANLinear(
                dimensions[index],
                dimensions[index + 1],
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=(-1.0, 1.0),
            )
            for index in range(len(dimensions) - 1)
        )
        self.intermediate_norms = nn.ModuleList(
            nn.LayerNorm(dimensions[index + 1]) for index in range(len(dimensions) - 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(torch.tanh(self.projection(x)))
        if self.head_type == "linear":
            return self.linear(x)
        for index, layer in enumerate(self.kan):
            x = layer(x)
            if index < len(self.intermediate_norms):
                x = self.dropout(torch.tanh(self.intermediate_norms[index](x)))
        return x

    def spline_regularization_loss(self) -> torch.Tensor:
        if not self.kan:
            return self.projection[0].weight.new_zeros(())
        penalties = [layer.scaled_spline_weight.abs().mean() for layer in self.kan]
        return torch.stack(penalties).mean()


class MobileNetV2KANPlus(nn.Module):
    """MobileNetV2-KAN+ with a modified backbone and bounded spline head."""

    _STAGE_CHANNELS = {6: 32, 13: 96, 17: 320}

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 256,
        bottleneck_dim: int = 512,
        dropout: float = 0.3,
        head_type: str = "kan",
        kan_layers: int = 2,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        use_cgfm: bool = True,
        cgfm_stages: Sequence[int] = (6, 13, 17),
        use_mixstyle: bool = True,
        mixstyle_probability: float = 0.5,
        mixstyle_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.features = mobilenet_v2(weights=None).features
        self.use_cgfm = bool(use_cgfm)
        self.use_mixstyle = bool(use_mixstyle)
        invalid = set(cgfm_stages).difference(self._STAGE_CHANNELS)
        if invalid:
            raise ValueError(f"Unsupported CGFM stages: {sorted(invalid)}")
        self.cgfm = nn.ModuleDict(
            {
                str(index): CrossScaleGatedFeatureModulation(self._STAGE_CHANNELS[index])
                for index in cgfm_stages
            }
        )
        self.mixstyle = MixStyle(mixstyle_probability, mixstyle_alpha)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = BoundedKANClassifier(
            in_features=1280,
            num_classes=num_classes,
            bottleneck_dim=bottleneck_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            head_type=head_type,
            kan_layers=kan_layers,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for index, block in enumerate(self.features):
            x = block(x)
            key = str(index)
            if self.use_cgfm and key in self.cgfm:
                x = self.cgfm[key](x)
            if self.use_mixstyle and index in {6, 13}:
                x = self.mixstyle(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.forward_features(x)).flatten(1)
        return self.classifier(x)

    def spline_regularization_loss(self) -> torch.Tensor:
        return self.classifier.spline_regularization_loss()


__all__ = [
    "BoundedKANClassifier",
    "CrossScaleGatedFeatureModulation",
    "MixStyle",
    "MobileNetV2KANPlus",
]
