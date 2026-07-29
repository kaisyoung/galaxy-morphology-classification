from __future__ import annotations

import torch.nn as nn
from torchvision import models

from src.baseline.train_baseline import (
    NUM_CLASSES,
    train_one,
)


def build_densenet121(
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    model = models.densenet121(
        weights=models.DenseNet121_Weights.IMAGENET1K_V1
    )

    model.classifier = nn.Linear(
        model.classifier.in_features,
        num_classes,
    )

    return model


def main():
    result = train_one(
        "densenet121",
        model=build_densenet121(),
    )

    metrics = result["test_metrics"]

    print(
        "\nDenseNet121 final test: "
        f"acc={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()