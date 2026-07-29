# Model Code Notes

Galaxy10 morphology classification. This file documents each member's model
files as they're added — baseline models, main models, and the shared
evaluation metrics. Each section below is written by the member who owns
that part; please add your own section rather than editing someone else's.

## Kaishi's part: baseline models, DenseNet121, shared metrics

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

## Yaoyu's part: ResNet50 & ConvNeXT-Nano (Zoobot + ImageNet variants)

Four model files, two architectures x two pretraining sources. All four
follow the same pattern as `densenet121.py`: `build_xxx()` only defines the
network, training reuses the shared `train_one()` unmodified (no class
weighting, no progressive unfreezing, no custom LR schedule — kept identical
to the baseline protocol for fair comparison).

### src/models/resnet50.py / src/models/convnext_nano.py
Galaxy-domain-pretrained versions, using Zoobot (Walmsley et al., 2021)
encoder weights instead of ImageNet. Loaded via `timm` from HuggingFace Hub
(`mwalmsley/zoobot-encoder-resnet50`, `mwalmsley/zoobot-encoder-convnext_nano`) —
downloads automatically on first run, needs internet access.

Run: `python -m src.models.resnet50` / `python -m src.models.convnext_nano`

### src/models/resnet50_imagenet.py / src/models/convnext_nano_imagenet.py
Same two architectures, standard ImageNet-pretrained weights (timm defaults)
instead of Zoobot. Added so there's an apples-to-apples ImageNet-basis
comparison against SimpleCNN/VGG16/DenseNet121 above — see
`outputs/results/baseline_reference.md` for why the Zoobot versions aren't
used in that primary comparison table (different pretraining source would
confound the architecture comparison).

Run: `python -m src.models.resnet50_imagenet` / `python -m src.models.convnext_nano_imagenet`

Results, per-class breakdowns, and the Zoobot-vs-ImageNet ablation are all in
`outputs/results/baseline_reference.md`.

## Notes
- Keep SEED/split/metrics consistent across the group (for better comparability)
- GPU architecture affects which PyTorch build you need — mismatches show up
  as CUDA/compatibility errors, not just slow training:
  - On Kaggle use T4, not P100 (P100 is sm_60, not compatible with the
    current PyTorch)
  - RTX 5080 (Blackwell, sm_120) needs PyTorch 2.7+ with CUDA 12.8 — install
    with `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`,
    the default `pip install torch` may not have Blackwell support
- Zoobot has no DenseNet encoder — the Zoobot-vs-ImageNet ablation only covers
  ResNet50/ConvNeXT-Nano, don't extrapolate that comparison to DenseNet121
- All of Yaoyu's runs (and the baseline re-runs in `baseline_reference.md`)
  were executed on one local RTX 5080, so hardware is held constant across
  every number in that file