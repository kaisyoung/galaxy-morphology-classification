from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
#Galaxy10 class names,index 0-9
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
def collect_predictions(model,loader, device):
    model.eval()
    y_true, y_pred = [], []
    for imgs,lbls in loader:
        imgs =imgs.to(device)
        logits =model(imgs)
        preds =logits.argmax(dim=1).cpu().numpy()
        y_pred.append(preds)
        y_true.append(np.asarray(lbls))
    return np.concatenate(y_true), np.concatenate(y_pred)

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true,y_pred),
        "macro_f1": f1_score(y_true, y_pred,average="macro"),
        "weighted_f1": f1_score(y_true, y_pred,average="weighted"),
    }

def evaluate(model,loader,device):
    y_true, y_pred = collect_predictions(model,loader,device)
    return compute_metrics(y_true, y_pred),y_true, y_pred

def full_report(y_true, y_pred,digits=3):
    return classification_report(
        y_true, y_pred, target_names=CLASS_NAMES,digits=digits,zero_division=0
    )

#confusion matrix(rows=true,cols=pred)
def get_confusion_matrix(y_true,y_pred):
    return confusion_matrix(y_true,y_pred,labels=list(range(len(CLASS_NAMES))))
