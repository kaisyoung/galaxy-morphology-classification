from __future__ import annotations

import timm
import torch.nn as nn

from src.baseline.train_baseline import NUM_CLASSES, train_one

# Zoobot's own recommended default encoder (galaxy-domain-pretrained,
# Walmsley et al.), same reasoning as resnet50.py.
ZOOBOT_CONVNEXT_NANO_HF_ID = "hf_hub:mwalmsley/zoobot-encoder-convnext_nano"


def build_convnext_nano(num_classes: int = NUM_CLASSES) -> nn.Module:
    return timm.create_model(ZOOBOT_CONVNEXT_NANO_HF_ID, pretrained=True, num_classes=num_classes)


def main():
    metrics = train_one("convnext_nano", model=build_convnext_nano())
    print(f"\nConvNeXT-Nano (Zoobot-pretrained, Yaoyu) final test: "
          f"acc={metrics['accuracy']:.4f}  macroF1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()