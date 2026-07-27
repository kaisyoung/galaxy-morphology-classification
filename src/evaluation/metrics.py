from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Galaxy10 class names, indexed from 0 to 9.
CLASS_NAMES = [
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Smooth",
    "Cigar-shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on w/o Bulge",
    "Edge-on w/ Bulge",
]


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Collect true and predicted class labels from a data loader."""

    model.eval()
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1).cpu().numpy()

        y_pred.append(predictions)
        y_true.append(np.asarray(labels))

    return np.concatenate(y_true), np.concatenate(y_pred)


def compute_metrics(y_true, y_pred):
    """Compute the shared overall classification metrics."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def compute_per_class_metrics(y_true, y_pred):
    """Compute precision, recall, F1 and support for each class."""

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )

    per_class_metrics = {}

    for index, class_name in enumerate(CLASS_NAMES):
        per_class_metrics[class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    return per_class_metrics


def evaluate(model, loader, device):
    """Evaluate one model and return metrics and predictions."""

    y_true, y_pred = collect_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred)

    return metrics, y_true, y_pred


def full_report(y_true, y_pred, digits=3):
    """Return the sklearn text classification report."""

    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=digits,
        zero_division=0,
    )


def get_confusion_matrix(y_true, y_pred):
    """Return confusion matrix where rows are true labels."""

    return confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
    )