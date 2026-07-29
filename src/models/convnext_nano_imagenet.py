from __future__ import annotations

import timm
import torch.nn as nn

from src.baseline.train_baseline import NUM_CLASSES, train_one

# Standard ImageNet-pretrained ConvNeXT-Nano (timm's default weights:
# convnext_nano.in12k_ft_in1k), kept alongside convnext_nano.py
# (Zoobot-pretrained) as an ablation/reference point -- same architecture,
# different pretraining source.


def build_convnext_nano_imagenet(num_classes: int = NUM_CLASSES) -> nn.Module:
    return timm.create_model("convnext_nano", pretrained=True, num_classes=num_classes)


def main():
    metrics = train_one("convnext_nano_imagenet", model=build_convnext_nano_imagenet())
    print(f"\nConvNeXT-Nano (ImageNet-pretrained, Yaoyu) final test: "
          f"acc={metrics['accuracy']:.4f}  macroF1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()