"""Model definitions: ResNet18 classifier with extensible factory."""

import logging
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

ModelName = Literal["resnet18", "resnet34", "resnet50"]


class ResNet18Classifier(nn.Module):
    """ResNet-18 adapted for grayscale chest X-Ray classification.

    The final fully-connected layer is replaced to match num_classes.
    Input images are expected as 3-channel RGB (grayscale converted to RGB).
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        """Initialize model.

        Args:
            num_classes: Number of output classes.
            pretrained: Load ImageNet pretrained weights.
        """
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, num_classes)
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNet34Classifier(nn.Module):
    """ResNet-34 classifier variant."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet34(weights=weights)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNet50Classifier(nn.Module):
    """ResNet-50 classifier variant."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet50(weights=weights)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_model(
    model_name: ModelName = "resnet18",
    num_classes: int = 6,
    pretrained: bool = True,
) -> nn.Module:
    """Factory function returning a classifier by name.

    Args:
        model_name: Architecture name ("resnet18", "resnet34", "resnet50").
        num_classes: Output class count.
        pretrained: Use ImageNet pretrained weights.

    Returns:
        PyTorch Module ready for training.
    """
    registry = {
        "resnet18": ResNet18Classifier,
        "resnet34": ResNet34Classifier,
        "resnet50": ResNet50Classifier,
    }
    if model_name not in registry:
        raise ValueError(f"Unknown model {model_name!r}. Available: {list(registry)}")

    model = registry[model_name](num_classes, pretrained)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Created %s with %d classes, %d parameters", model_name, num_classes, n_params)
    return model
