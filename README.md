# Galaxy Morphology Classification

Deep-learning pipeline that classifies galaxy images into **10 morphology
classes** on the **Galaxy10 DECaLS** dataset (17,736 RGB images). The project
compares **seven CNN configurations** — from a from-scratch baseline to modern
architectures with domain-specific pretraining — under one shared, controlled
training and evaluation protocol.

> UNSW COMP9444 (Neural Networks & Deep Learning) group project, Term 2 2026.
> This repository is a copy I maintain for my portfolio. See
> [My contribution](#my-contribution) for the parts I wrote; other components
> were built by teammates (credited below).

## Key results

Seven models trained under an identical protocol (stratified 80/10/10 split,
seed 42, AdamW, 15 epochs, best checkpoint by validation Macro-F1).

| Model | Pretraining | Macro-F1 | Params |
|-------|-------------|----------|--------|
| ConvNeXT-Nano | Zoobot | **0.860** | 15.0M |
| ResNet50 | Zoobot | 0.851 | 23.5M |
| ConvNeXT-Nano | ImageNet | 0.822 | 15.0M |
| **DenseNet121** | ImageNet | **0.806** | **7.0M** |
| VGG16 | ImageNet | 0.810 | 134M |
| ResNet50 | ImageNet | 0.779 | 23.5M |
| SimpleCNN | none (scratch) | 0.377 | 0.4M |

**Findings:** (1) transfer learning hugely outperforms training from scratch
(0.38 → 0.80+); (2) **domain-specific Zoobot pretraining beats generic ImageNet**
for both architectures — the pretraining source matters more than the
architecture; (3) **DenseNet121 matches VGG16's accuracy with ~19× fewer
parameters** — a parameter-efficiency sweet spot.

## My contribution

I (Kaishi Yang) implemented:

- **Baseline models** — `SimpleCNN` (from scratch, lower bound) and `VGG16`
  (ImageNet transfer-learning reference) in `src/baseline/train_baseline.py`.
- **Main model** — `DenseNet121` in `src/models/densenet121.py`.
- **Shared training framework** — `train_one()`, the single training/evaluation
  protocol reused unchanged by all seven models (fixed split, seed, optimizer,
  epochs, and best-checkpoint selection by validation Macro-F1). This keeps every
  model's training identical and makes the cross-architecture comparison and the
  pretraining ablation fair.
- **Shared evaluation module** — `src/evaluation/metrics.py` (Accuracy,
  Macro-F1, per-class report, confusion matrix), used by every model, with
  Macro-F1 as the primary metric for the class-imbalanced dataset.

## Tech stack

Python · PyTorch · torchvision · timm · scikit-learn · astroNN · matplotlib ·
Grad-CAM. Pretrained weights from torchvision/timm (ImageNet) and Zoobot
(Walmsley et al.) via HuggingFace Hub.

## Repository structure

```
src/preprocessing/   data loading, stratified split, transforms
src/baseline/        SimpleCNN + VGG16 + shared train_one() protocol   (mine)
src/models/          DenseNet121 (mine) + ResNet50 / ConvNeXT-Nano variants
src/evaluation/      metrics.py (mine) + comparison, confusion matrix, Grad-CAM
notebooks/           EDA + integrated project notebook
outputs/             figures, summary results
```

## Running

```bash
pip install -r requirements.txt
python -m src.baseline.train_baseline     # SimpleCNN + VGG16
python -m src.models.densenet121          # DenseNet121
```
(Galaxy10 downloads automatically via astroNN on first run.)

## Team & attribution

A 5-person group project. My part is listed under
[My contribution](#my-contribution); teammates built the data pipeline, the
ResNet50/ConvNeXT-Nano + Zoobot models, the evaluation/visualization pipeline,
and the final notebook. Public architectures and pretrained weights are used as
building blocks, not claimed as original work.
