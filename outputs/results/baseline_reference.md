# 这是 Colab 验证跑结果（仅供Yaoyu参考）不作为最终报告数据,最后需要以统一跑为准
# Baseline & DenseNet121

> Colab validation run. Final numbers will come from Member C's single consolidated run.

Environment: Google Colab (Tesla T4), single run. Protocol: shared `train_one`
(80/10/10 stratified split, `random_state=42`, 15 epochs, AdamW lr=1e-4,
best checkpoint by val macro-F1). Metrics: shared `src/evaluation/metrics.py`.

## Test-set results (1774 images)

| Model        | Owner | Params | test_acc | test_macroF1 | Role                       |
|--------------|-------|--------|----------|--------------|----------------------------|
| SimpleCNN    | Kaishi | ~0.4M  | 0.4645   | 0.3827       | Lower-bound (from scratch) |
| VGG16        | Kaishi | ~138M  | 0.8405   | 0.8149       | Transfer-learning ref      |
| DenseNet121  | Kaishi | ~8M    | 0.8140   | 0.8048       | main model                 |

(ResNet50, ViT-B/16 — Member C — to be appended here in the same format.)

## Key findings

1. **Transfer learning dominates**: from-scratch SimpleCNN reaches only 0.38
   macro-F1 after 15 epochs (still underfitting); ImageNet-pretrained models
   exceed 0.68 after a single epoch and plateau ~0.80.
2. **Parameter efficiency**: DenseNet121 attains 98.7% of VGG16's macro-F1
   using ~6% of the parameters (8M vs 138M) — a strong efficiency argument.
3. **Class imbalance**: the smallest class (Cigar-shaped Smooth, 34 test
   samples) goes from F1=0.000 (SimpleCNN) to 0.83/0.73 (DenseNet/VGG16);
   pretrained features rescue minority classes.
4. **Intrinsic difficulty**: all models struggle most on Disturbed and on
   Unbarred Loose vs Tight Spiral — morphologically similar categories that are
   consistently confused in the confusion matrices.

## Per-class F1 (test)

| Class                  | SimpleCNN | VGG16 | DenseNet121 |
|------------------------|-----------|-------|-------------|
| Disturbed              | 0.134     | 0.517 | 0.524       |
| Merging                | 0.169     | 0.881 | 0.863       |
| Round Smooth           | 0.541     | 0.905 | 0.894       |
| In-between Smooth      | 0.451     | 0.907 | 0.907       |
| Cigar-shaped Smooth    | 0.000     | 0.729 | 0.829       |
| Barred Spiral          | 0.366     | 0.881 | 0.830       |
| Unbarred Tight Spiral  | 0.412     | 0.772 | 0.707       |
| Unbarred Loose Spiral  | 0.476     | 0.752 | 0.671       |
| Edge-on w/o Bulge      | 0.568     | 0.892 | 0.901       |
| Edge-on w/ Bulge       | 0.709     | 0.913 | 0.922       |
