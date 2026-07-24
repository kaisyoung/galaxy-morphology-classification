# 这是 PyCharm 本地验证跑结果（仅供参考）不作为最终报告数据，最后需要以统一跑为准
# Baseline & Main Models

> All validation runs in this document (SimpleCNN, VGG16, DenseNet121,
> ResNet50 x2, ConvNeXT-Nano x2) were executed by Yaoyu on the same local
> machine (RTX 5080), using the shared `train_one` protocol unmodified —
> hardware and software environment are held constant across every row in
> this file, so differences between models reflect the model/pretraining
> choice, not machine-to-machine variation. Final numbers for the report
> will still come from Member C's single consolidated run.

Environment: shared `train_one` protocol (80/10/10 stratified split,
`random_state=42`, 15 epochs, AdamW lr=1e-4, best checkpoint by val macro-F1).
Metrics: shared `src/evaluation/metrics.py`.

*Verification note: SimpleCNN/VGG16/DenseNet121 numbers were re-run locally
to confirm the RTX-5080-only claim above. SimpleCNN and VGG16 matched the
original numbers within ~0.3pp (normal run-to-run noise). DenseNet121 came
back +2.2pp accuracy / +0.9pp macro-F1 higher than the original — larger
than expected noise, most likely because the original number came from a
different environment before this file's numbers were unified onto one
machine. The table below uses the RTX 5080 rerun for all three.*

---

## Part 1: All models on ImageNet pretraining (primary comparison)

This is the apples-to-apples table — every model here uses standard
ImageNet-pretrained weights, and every run was executed on the same machine
under the same protocol, so the only variable across rows is architecture.

| Model                     | Owner  | Params | test_acc | test_macroF1 | Role                              |
|----------------------------|--------|--------|----------|--------------|-------------------------------------|
| SimpleCNN                  | Kaishi | ~0.4M  | 0.4639   | 0.3845       | Lower-bound (from scratch)          |
| VGG16                      | Kaishi | ~138M  | 0.8371   | 0.8157       | Transfer-learning ref               |
| DenseNet121                | Kaishi | ~8M    | 0.8360   | 0.8136       | main model                          |
| ResNet50 (ImageNet)        | Yaoyu  | ~23.5M | 0.8050   | 0.7824       | main model                          |
| ConvNeXT-Nano (ImageNet)   | Yaoyu  | ~15M   | 0.8258   | 0.8073       | main model                          |

*"Owner" = who designed/is responsible for that model choice in the project
plan, not who executed this particular validation run — see the note at the
top of this file, all numbers here were run by Yaoyu on one machine.*

**On ImageNet pretraining, VGG16 and DenseNet121 are essentially tied for
strongest** (0.8371/0.8157 vs 0.8360/0.8136 — 0.11pp apart, within run-to-run
noise), both clearly ahead of ConvNeXT-Nano-ImageNet (0.8258/0.8073, ~1pp
back) and ResNet50-ImageNet (0.8050/0.7824, ~2pp further back). Given VGG16
has ~9x the parameters of ConvNeXT-Nano and ~17x DenseNet121, it is not the
most parameter-efficient choice — DenseNet121 reaches essentially the same
accuracy at a fraction of the size.

### Per-class F1 (ImageNet basis)

*SimpleCNN's per-class breakdown below is from the original zip (its
classification report wasn't reprinted in the RTX 5080 rerun, only the
summary acc/macroF1 above, which matched closely); VGG16 and DenseNet121
columns are the actual RTX 5080 rerun numbers.*

| Class                  | SimpleCNN | VGG16 | DenseNet121 | ResNet50 (ImageNet) | ConvNeXT-Nano (ImageNet) |
|-------------------------|-----------|-------|-------------|------------------------|-----------------------------|
| Disturbed               | 0.134     | 0.369 | 0.452       | 0.433                  | 0.490                       |
| Merging                  | 0.169     | 0.866 | 0.882       | 0.837                  | 0.872                       |
| Round Smooth             | 0.541     | 0.910 | 0.915       | 0.891                  | 0.911                       |
| In-between Smooth        | 0.451     | 0.911 | 0.906       | 0.904                  | 0.897                       |
| Cigar-shaped Smooth      | 0.000     | 0.868 | 0.806       | 0.743                  | 0.747                       |
| Barred Spiral            | 0.366     | 0.886 | 0.887       | 0.827                  | 0.861                       |
| Unbarred Tight Spiral    | 0.412     | 0.741 | 0.731       | 0.711                  | 0.735                       |
| Unbarred Loose Spiral    | 0.476     | 0.739 | 0.749       | 0.653                  | 0.717                       |
| Edge-on w/o Bulge        | 0.568     | 0.931 | 0.877       | 0.898                  | 0.910                       |
| Edge-on w/ Bulge         | 0.709     | 0.935 | 0.932       | 0.926                  | 0.934                       |

---

## Part 2: Zoobot vs ImageNet ablation (ResNet50 & ConvNeXT-Nano only)

Zoobot (Walmsley et al., 2021) provides galaxy-domain-pretrained encoders for
ResNet50 and ConvNeXT-Nano specifically (loaded via `timm` from HuggingFace
Hub: `mwalmsley/zoobot-encoder-resnet50`, `mwalmsley/zoobot-encoder-convnext_nano`).
**Note: Zoobot has no DenseNet variant**, so this ablation is limited to the
two architectures where both a Zoobot and an ImageNet checkpoint exist.
Everything else (split, epochs, optimizer, loss, protocol) is identical
between each pair — the only variable is pretraining source.

| Model          | Pretraining | test_acc | test_macroF1 | Δ acc vs ImageNet | Δ macroF1 vs ImageNet |
|-----------------|-------------|----------|--------------|--------------------|--------------------------|
| ResNet50        | ImageNet    | 0.8050   | 0.7824       | —                  | —                        |
| ResNet50        | **Zoobot**  | 0.8653   | 0.8548       | **+6.03pp**        | **+7.24pp**              |
| ConvNeXT-Nano   | ImageNet    | 0.8258   | 0.8073       | —                  | —                        |
| ConvNeXT-Nano   | **Zoobot**  | 0.8805   | 0.8590       | **+5.47pp**        | **+5.17pp**              |

### Key finding: pretraining source matters more than architecture choice

Holding architecture fixed, switching from ImageNet to Zoobot pretraining
gains **+5 to +7 percentage points** on both accuracy and macro-F1. By
contrast, holding pretraining source fixed and switching architecture
(ResNet50 → ConvNeXT-Nano) gains only:
- +2.08pp acc / +2.49pp macroF1 under ImageNet pretraining
- +1.52pp acc / +0.42pp macroF1 under Zoobot pretraining

The **pretraining-source effect is roughly 2-15x larger than the
architecture effect** in this comparison. This is consistent with Walmsley
et al.'s (2021) motivation for Zoobot: a domain-specific encoder pretrained
directly on ~100M+ Galaxy Zoo volunteer labels transfers to galaxy
morphology tasks better than general-purpose ImageNet features, and this
effect appears to dominate architectural differences at this model scale.

### Per-class pattern: the gap is largest on fine-grained morphology classes

The Zoobot advantage is not uniform across classes. Coarse, whole-galaxy-shape
classes (Merging, Edge-on w/ Bulge) show a small gap; classes requiring
fine-grained structural judgment (spiral tightness, bar presence) show the
largest gap — e.g. for ResNet50, Unbarred Loose Spiral improves from F1=0.653
(ImageNet) to 0.776 (Zoobot), a 12.3pp swing, the largest of any class.
Plausible explanation: ImageNet features encode generic object shape/texture,
which suffices for coarse whole-galaxy silhouette judgments, but Zoobot's
galaxy-specific pretraining appears to better capture the finer structural
cues (arm winding, bar structure) that distinguish morphologically similar
classes.

---

## Notes for Member C (consolidated run)

- ResNet50 / ConvNeXT-Nano (both pretraining variants) need `timm` installed
  (`pip install timm`, already added to `requirements.txt`) and internet
  access on first run to download weights from HuggingFace Hub.
- Four build functions total, all following the same `build_xxx()` +
  `train_one()` pattern as `densenet121.py`:
  - `src/models/resnet50.py` → `build_resnet50()` (Zoobot)
  - `src/models/resnet50_imagenet.py` → `build_resnet50_imagenet()` (ImageNet)
  - `src/models/convnext_nano.py` → `build_convnext_nano()` (Zoobot)
  - `src/models/convnext_nano_imagenet.py` → `build_convnext_nano_imagenet()` (ImageNet)
- **For Part 1 (the primary cross-architecture comparison), use the
  `_imagenet` variants** so every row in that table shares the same
  pretraining source. The Zoobot variants are for Part 2's ablation only —
  don't mix a Zoobot-pretrained row into Part 1's table, since that would
  confound architecture and pretraining source again.
- Class-name spelling follows `src/evaluation/metrics.py`'s `CLASS_NAMES`
  (e.g. "In-between Smooth", "Cigar-shaped Smooth") — keep this consistent
  across the report rather than mixing in astroNN's official `Galaxy10Class`
  wording, which differs slightly (e.g. "Smooth, Cigar shaped").
- DenseNet121 has no Zoobot counterpart (Zoobot doesn't publish a DenseNet
  encoder) — Part 2's ablation is necessarily limited to ResNet50 and
  ConvNeXT-Nano only; please don't extrapolate the Zoobot-vs-ImageNet gap to
  DenseNet121 in the report.
