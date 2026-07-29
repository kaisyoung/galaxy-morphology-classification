#Baseline: SimpleCNN (from scratch) + VGG16 (transfer).

from __future__ import annotations
import platform
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from src.preprocessing.dataloader import build_dataloaders
from src.evaluation.metrics import evaluate, full_report, get_confusion_matrix

#shared config --keep consistent
SEED =42
BATCH_SIZE =64
EPOCHS =15
LR =1e-4
NUM_CLASSES =10
IMG_SIZE =224
NUM_WORKERS =0 if platform.system() =="Darwin" else 2   # 0 on macOS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

IMAGENET_MEAN =[0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


#minimal transforms
def build_transforms():
    train_tf = transforms.Compose([
        transforms.ToTensor(),                       #uint8 to float [0,1]
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 32), block(32, 64),block(64, 128),block(128, 256)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.head(self.features(x))


def build_model(name: str) -> nn.Module:
    name = name.lower()
    if name =="simplecnn":
        return SimpleCNN()
    if name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, NUM_CLASSES)
        return m
    raise ValueError(f"Unknown model: {name}")


#train +evaluate one model with the share protocol
#pass `model` to run you own architecture here & use baseline model
def train_one(model_name: str, model: nn.Module | None = None):
    train_tf, eval_tf = build_transforms()

    #don't re-split here
    train_loader, val_loader,test_loader = build_dataloaders(
        batch_size=BATCH_SIZE,
        train_ratio=0.8, val_ratio=0.1,test_ratio=0.1,
        random_state=SEED,
        num_workers=NUM_WORKERS,
        train_transform=train_tf,
        val_transform=eval_tf,
        test_transform=eval_tf,
    )

    if model is None:
        model = build_model(model_name)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_f1, best_w = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = 0.0
        for imgs, lbls in train_loader:
            imgs =imgs.to(DEVICE)
            lbls =lbls.to(DEVICE)
            optimizer.zero_grad()
            loss =criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            run_loss +=loss.item() * imgs.size(0)

        val_m, _, _ =evaluate(model, val_loader, DEVICE)
        print(f"[{model_name}] epoch {epoch:02d} "
              f"train_loss={run_loss / len(train_loader.dataset):.4f} "
              f"val_acc={val_m['accuracy']:.4f} "
              f"val_macroF1={val_m['macro_f1']:.4f}")

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            best_w = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


    model.load_state_dict(best_w)
    test_m, y_true,y_pred = evaluate(model,test_loader, DEVICE)
    print(f"\n=== {model_name} TEST  "
          f"acc={test_m['accuracy']:.4f}  "
          f"macroF1={test_m['macro_f1']:.4f} ===")
    print(full_report(y_true, y_pred))
    print("Confusion matrix:\n", get_confusion_matrix(y_true, y_pred))
    return test_m


def main():
    results ={}
    for name in ["simplecnn", "vgg16"]:
        results[name] = train_one(name)
    print("\n================ BASELINE SUMMARY ================")
    print(f"{'model':<14}{'test_acc':>10}{'test_macroF1':>14}")
    for name, m in results.items():
        print(f"{name:<14}{m['accuracy']:>10.4f}{m['macro_f1']:>14.4f}")


if __name__ == "__main__":
    main()
