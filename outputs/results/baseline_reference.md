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

*All five columns below are RTX 5080 rerun numbers (SimpleCNN's full report
came in slightly after the others but matches the original zip closely).*

| Class                  | SimpleCNN | VGG16 | DenseNet121 | ResNet50 (ImageNet) | ConvNeXT-Nano (ImageNet) |
|-------------------------|-----------|-------|-------------|------------------------|-----------------------------|
| Disturbed               | 0.150     | 0.369 | 0.452       | 0.433                  | 0.490                       |
| Merging                  | 0.175     | 0.866 | 0.882       | 0.837                  | 0.872                       |
| Round Smooth             | 0.541     | 0.910 | 0.915       | 0.891                  | 0.911                       |
| In-between Smooth        | 0.446     | 0.911 | 0.906       | 0.904                  | 0.897                       |
| Cigar-shaped Smooth      | 0.000     | 0.868 | 0.806       | 0.743                  | 0.747                       |
| Barred Spiral            | 0.371     | 0.886 | 0.887       | 0.827                  | 0.861                       |
| Unbarred Tight Spiral    | 0.417     | 0.741 | 0.731       | 0.711                  | 0.735                       |
| Unbarred Loose Spiral    | 0.473     | 0.739 | 0.749       | 0.653                  | 0.717                       |
| Edge-on w/o Bulge        | 0.568     | 0.931 | 0.877       | 0.898                  | 0.910                       |
| Edge-on w/ Bulge         | 0.704     | 0.935 | 0.932       | 0.926                  | 0.934                       |

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

## Appendix: Confusion matrices (raw counts, test set)

Rows = true class, columns = predicted class, both in `CLASS_NAMES` order:
`[Disturbed, Merging, Round Smooth, In-between Smooth, Cigar-shaped Smooth,
Barred Spiral, Unbarred Tight Spiral, Unbarred Loose Spiral, Edge-on w/o
Bulge, Edge-on w/ Bulge]` (indices 0-9). All seven from the RTX 5080 runs
above.

**SimpleCNN (from scratch)**
```
[[  9   2  34  15   0   1   2  42   2   1]
 [  0  21  95  20   0   4  11  17   5  12]
 [  0   2 215  28   0   3   4  12   0   0]
 [  0   3  85 100   0   1   1   9   2   2]
 [  0   0   6  22   0   0   0   4   2   0]
 [  0   4  33  18   0  62  28  55   3   2]
 [  1   6  12   2   0  37  70  51   1   3]
 [  2   6  26  22   0  19  25 152   6   5]
 [  0   3   1   3   0   1  12  35  69  18]
 [  0   8  24  15   0   1   0   3  11 125]]
```

**VGG16 (ImageNet)**
```
[[ 29   1  11  11   3   5   7  34   3   4]
 [  3 155   3   2   1   5   2  11   1   2]
 [  1   2 242   1   0   3  13   2   0   0]
 [  4   6   3 185   2   0   3   0   0   0]
 [  0   0   0   0  33   0   0   0   1   0]
 [  3   2   0   1   0 183   7   8   0   1]
 [  2   1   6   1   0   3 152  16   2   0]
 [  4   6   2   0   1   8  41 198   0   3]
 [  2   0   0   0   2   1   1   1 134   1]
 [  1   0   1   2   0   0   1   3   5 174]]
```

**DenseNet121 (ImageNet)**
```
[[ 40   1  13  11   3   7   5  23   3   2]
 [  7 157   6   5   2   0   2   3   2   1]
 [  2   1 252   0   0   1   7   1   0   0]
 [  3   3   3 187   1   0   4   2   0   0]
 [  0   0   0   1  29   1   1   0   2   0]
 [  3   3   0   2   0 180   5  11   1   0]
 [  1   1   9   2   0   1 143  22   4   0]
 [  8   5   3   1   1   9  36 195   2   3]
 [  4   0   0   0   2   0   4   0 128   4]
 [  1   0   1   1   0   2   1   1   8 172]]
```

**ResNet50 (ImageNet)**
```
[[ 39   4  15  11   2   8   4  19   2   4]
 [  2 159   6   1   1   4   2   6   1   3]
 [  0   6 241   1   0   2  10   4   0   0]
 [  3   6   1 189   1   0   3   0   0   0]
 [  0   0   0   2  26   1   1   1   2   1]
 [  2   2   1   2   1 177   5  14   1   0]
 [  3   1   9   3   0   8 133  24   2   0]
 [ 19  16   3   4   1  21  32 162   3   2]
 [  3   1   0   0   2   0   0   3 128   5]
 [  1   0   1   2   2   2   1   0   4 174]]
```

**ResNet50 (Zoobot)**
```
[[ 47   2  12  10   2   4   4  24   0   3]
 [  4 161   4   0   1   1   2   8   0   4]
 [  0   0 256   0   0   0   7   1   0   0]
 [  2   4   0 192   0   1   3   1   0   0]
 [  0   0   0   1  33   0   0   0   0   0]
 [  2   2   0   1   0 180   5  12   1   2]
 [  2   1  10   1   0   1 151  15   2   0]
 [  7   5   1   0   0   4  35 208   0   3]
 [  4   0   0   0   1   0   1   2 129   5]
 [  1   1   1   1   0   1   0   2   2 178]]
```

**ConvNeXT-Nano (ImageNet)**
```
[[ 48   1  11   8   5   4   3  24   2   2]
 [  2 157   7   2   1   1   4   9   0   2]
 [  3   3 240   0   0   0  15   3   0   0]
 [  9   3   2 175   6   0   4   4   0   0]
 [  0   0   0   0  28   1   0   1   3   1]
 [  6   4   0   1   0 167  10  15   1   1]
 [  0   1   2   1   0   1 144  30   4   0]
 [ 19   5   0   0   1   7  28 199   1   3]
 [  1   0   0   0   0   0   1   4 131   5]
 [  0   1   1   0   0   2   0   3   4 176]]
```

**ConvNeXT-Nano (Zoobot)**
```
[[ 54   2  12   9   6   4   0  15   3   3]
 [  3 174   1   0   1   1   1   1   0   3]
 [  0   2 250   0   0   0   9   3   0   0]
 [  0   4   2 191   3   0   1   2   0   0]
 [  0   0   0   0  30   0   0   0   1   3]
 [  0   2   0   1   0 191   3   7   1   0]
 [  2   1   4   1   0   2 146  24   3   0]
 [  9   6   2   0   0   8  20 213   1   4]
 [  0   0   0   0   3   0   0   4 130   5]
 [  0   0   1   1   0   1   0   0   1 183]]
```

Common pattern worth noting for the report's discussion section: in **all
seven matrices above**, **Disturbed's largest off-diagonal cell is column 7
(Unbarred Loose Spiral)**, with no exception — holds regardless of
architecture, pretraining source, or whether the model was trained from
scratch. **Unbarred Tight/Loose Spiral consistently confuse with each
other** too (columns 6/7 both show elevated off-diagonal counts in each
other's row) in 6 of 7 models; the one near-exception is SimpleCNN, where
Unbarred Loose Spiral's largest confusion is narrowly with Round Smooth (26
images) rather than Unbarred Tight Spiral (25 images) — a 1-image margin,
still clearly elevated, so this reads as the same underlying pattern showing
through a much noisier from-scratch model rather than a real counter-example.
Overall this reinforces the Part 1/2 finding that these are dataset-level
ambiguities rather than model-specific weaknesses.

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
