from __future__ import annotations

import timm
import torch.nn as nn

from src.baseline.train_baseline import NUM_CLASSES, train_one

# Standard ImageNet-pretrained ResNet50 (timm's default weights: resnet50.a1_in1k),
# kept alongside resnet50.py (Zoobot-pretrained) as an ablation/reference
# point -- same architecture, different pretraining source. Also happens to
# match train_one()'s hardcoded ImageNet-mean/std transform exactly, unlike
# the Zoobot version.


def build_resnet50_imagenet(num_classes: int = NUM_CLASSES) -> nn.Module:
    return timm.create_model("resnet50", pretrained=True, num_classes=num_classes)


def main():
    result = train_one(
        "resnet50_imagenet",
        model=build_resnet50_imagenet(),
    )

    metrics = result["test_metrics"]

    print(
        "\nResNet50 ImageNet final test: "
        f"acc={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )

if __name__ == "__main__":
    main()