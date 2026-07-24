from __future__ import annotations

import timm
import torch.nn as nn

from src.baseline.train_baseline import NUM_CLASSES, train_one

# Zoobot's galaxy-domain-pretrained ResNet50 encoder (Walmsley et al.),
# hosted on HuggingFace Hub. Zoobot encoders are published as plain timm
# models, so no custom loading code is needed.
ZOOBOT_RESNET50_HF_ID = "hf_hub:mwalmsley/zoobot-encoder-resnet50"


def build_resnet50(num_classes: int = NUM_CLASSES) -> nn.Module:
    return timm.create_model(ZOOBOT_RESNET50_HF_ID, pretrained=True, num_classes=num_classes)


def main():
    metrics = train_one("resnet50", model=build_resnet50())
    print(f"\nResNet50 (Zoobot-pretrained, Yaoyu) final test: "
          f"acc={metrics['accuracy']:.4f}  macroF1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()