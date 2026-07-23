# Baseline Code Notes

Galaxy10 morphology classification. This covers Member Kaishi's part:
the baseline models, the main model, and the shared evaluation metrics.

### src/baseline/train_baseline.py
Baseline: SimpleCNN (from scratch) + VGG16 (transfer learning).
Data comes from Jiakai's `build_dataloaders()` and scoring use the shared metrics,
so every model train and compare in the same way. Resize + ImageNet-norm
transforms are added here because the default transform is only `ToTensor`.
`train_one()` is the shared training/eval protocol that reused by all main models.

Run: `python -m src.baseline.train_baseline`

### src/models/densenet121.py
DenseNet121 — main model. It only define the network, training reuse
`train_one()` from the baseline.
Same pattern for other models: `build_xxx()` then
`train_one("name", model=build_xxx())`.

Run: `python -m src.models.densenet121`

### src/evaluation/metrics.py
Shared evaluation metrics — import this so every model are scored in the same way.
Macro-F1 is the main metric (the dataset is class-imbalanced); accuracy and a
per-class report is also kept.

## Notes
- Keep SEED/split/metrics consistent across the group (for better comparability)
- On Kaggle use T4, not P100 (P100 sm_60 is not compatible with the current PyTorch)