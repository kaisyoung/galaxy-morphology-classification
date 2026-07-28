#Baseline: SimpleCNN (from scratch) + VGG16 (transfer).

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from src.evaluation.metrics import (
    compute_per_class_metrics,
    evaluate,
    full_report,
    get_confusion_matrix,
)
from src.preprocessing.dataloader import build_dataloaders

SEED =42
BATCH_SIZE =64
EPOCHS =15
LR =1e-4
NUM_CLASSES =10
IMG_SIZE =224
SYSTEM = platform.system()
NUM_WORKERS = 0 if SYSTEM in {"Windows", "Darwin"} else 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "outputs" / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

IMAGENET_MEAN =[0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


#minimal transforms
def build_transforms():
    train_tf = transforms.Compose([
        transforms.ToTensor(), 
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


#train and evaluate one model with the share protocol
def train_one(
    model_name: str,
    model: nn.Module | None = None,
):

    train_transform, eval_transform = build_transforms()

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=BATCH_SIZE,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=SEED,
        num_workers=NUM_WORKERS,
        train_transform=train_transform,
        val_transform=eval_transform,
        test_transform=eval_transform,
    )

    if model is None:
        model = build_model(model_name)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
    }

    best_macro_f1 = -1.0
    best_state_dict = None
    best_epoch = 0

    training_start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        validation_metrics, _, _ = evaluate(
            model,
            val_loader,
            DEVICE,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_accuracy"].append(
            validation_metrics["accuracy"]
        )
        history["val_macro_f1"].append(
            validation_metrics["macro_f1"]
        )
        history["val_weighted_f1"].append(
            validation_metrics["weighted_f1"]
        )

        print(
            f"[{model_name}] "
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"val_acc={validation_metrics['accuracy']:.4f} "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f}"
        )

        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch

            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    training_time_seconds = time.perf_counter() - training_start_time

    if best_state_dict is None:
        raise RuntimeError(
            f"No valid checkpoint was produced for {model_name}."
        )

    model.load_state_dict(best_state_dict)

    checkpoint_path = CHECKPOINT_DIR / f"{model_name}.pt"
    torch.save(best_state_dict, checkpoint_path)

    inference_start_time = time.perf_counter()

    test_metrics, y_true, y_pred = evaluate(
        model,
        test_loader,
        DEVICE,
    )

    inference_time_seconds = (
        time.perf_counter() - inference_start_time
    )

    per_class_metrics = compute_per_class_metrics(
        y_true,
        y_pred,
    )

    confusion = get_confusion_matrix(
        y_true,
        y_pred,
    )

    result = {
        "model_name": model_name,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "image_size": IMG_SIZE,
        "device": str(DEVICE),
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(
            trainable_parameter_count
        ),
        "best_epoch": int(best_epoch),
        "best_validation_macro_f1": float(
            best_macro_f1
        ),
        "training_time_seconds": float(
            training_time_seconds
        ),
        "inference_time_seconds": float(
            inference_time_seconds
        ),
        "test_sample_count": int(len(y_true)),
        "test_metrics": test_metrics,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion.tolist(),
        "history": history,
    }

    result_path = RESULT_DIR / f"{model_name}.json"

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as result_file:
        json.dump(
            result,
            result_file,
            indent=2,
        )

    prediction_path = (
        RESULT_DIR / f"{model_name}_predictions.npz"
    )

    np.savez(
        prediction_path,
        y_true=y_true,
        y_pred=y_pred,
    )

    print(
        f"\n=== {model_name} TEST ===\n"
        f"accuracy={test_metrics['accuracy']:.4f}\n"
        f"macro_f1={test_metrics['macro_f1']:.4f}\n"
        f"weighted_f1={test_metrics['weighted_f1']:.4f}\n"
        f"best_epoch={best_epoch}\n"
        f"parameters={parameter_count:,}\n"
        f"training_time={training_time_seconds:.2f}s\n"
    )

    print(full_report(y_true, y_pred))
    print("Confusion matrix:\n", confusion)
    print(f"Saved result to: {result_path}")
    print(f"Saved checkpoint to: {checkpoint_path}")

    return result


def main():
    results = {}

    for name in ["simplecnn", "vgg16"]:
        results[name] = train_one(name)

    print("\n================ BASELINE SUMMARY ================")

    print(
        f"{'model':<14}"
        f"{'test_acc':>12}"
        f"{'test_macro_f1':>16}"
    )

    for name, result in results.items():
        test_metrics = result["test_metrics"]

        print(
            f"{name:<14}"
            f"{test_metrics['accuracy']:>12.4f}"
            f"{test_metrics['macro_f1']:>16.4f}"
        )


if __name__ == "__main__":
    main()
