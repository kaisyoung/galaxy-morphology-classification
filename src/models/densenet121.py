from __future__ import annotations
import torch.nn as nn
from torchvision import models
from src.baseline.train_baseline import NUM_CLASSES,train_one


def build_densenet121(num_classes: int = NUM_CLASSES) -> nn.Module:
    m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m

def main():
    metrics = train_one("densenet121",model=build_densenet121())
    print(f"\nDenseNet121 (Kaishi) final test: "
          f"acc={metrics['accuracy']:.4f}  macroF1={metrics['macro_f1']:.4f}")

if __name__ =="__main__":
    main()
