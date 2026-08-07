# Steel Surface Defect Detection: Classification and Segmentation on the Severstal Dataset

**Authors:** _[TEAM MEMBER NAMES — to be filled in before submission]_

**Course:** ADSP 32023 Advanced Computer Vision — Final Project

---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction and Dataset](#1-introduction-and-dataset)
3. [Exploratory Data Analysis](#2-exploratory-data-analysis)
4. [Problem 1 — Defect Classification](#3-problem-1--defect-classification)
5. [Problem 2 — Defect Segmentation](#4-problem-2--defect-segmentation)
6. [Model Operations](#5-model-operations)
7. [Conclusion](#6-conclusion)
8. [References](#references)

---

## Abstract

We study automated inspection of hot-rolled steel surfaces using the Severstal dataset of 12,568
labelled 1600×256 grayscale images across four defect classes. From one dataset we pose two
distinct cognitive problems: multi-label classification (which of four defect types is present)
and pixel-level segmentation (where each defect is). For each problem we train a champion and a
challenger on an identical recipe so the comparison isolates the model rather than the training
setup. In classification, EfficientNet-B2 (challenger) reaches a macro-F1 of 0.934 against
ResNet-50's 0.904; a loss-function ablation shows plain BCE beats focal loss and class
positive-weighting for this label distribution. In segmentation, DeepLabV3+ (challenger) reaches a
defect-only Dice of 0.725 against U-Net's 0.705 on a shared ResNet-34 encoder, and wins on every
sub-metric we report. The segmentation work also surfaces a measurement trap specific to this
dataset: under the competition's own all-pairs Dice convention a model that predicts nothing
scores 0.859, higher than either trained model, because empty-versus-empty pairs are scored as
perfect. We therefore report defect-only metrics as the headline, false-positive behaviour
separately, and an operating-point sensitivity curve, because the right precision/recall trade for
an inspection line is a business decision rather than a number to tune on the validation set.

---

## 1. Introduction and Dataset

Surface-defect inspection is a natural computer-vision problem: the defects are visible, the
labels are physical, and both "is there a defect" and "where is it" matter to a production line.
The Severstal dataset provides 12,568 labelled grayscale images, each a 1600×256 strip of rolled
steel, annotated with run-length-encoded masks for four defect classes.

The dataset supports two genuinely different cognitive problems, which is what the project calls
for:

- **Classification** — a per-image, multi-label judgement of which of the four defect classes are
  present. This is the triage question on an inspection line.
- **Segmentation** — a per-pixel mask of each defect. This is the localisation question that
  drives measurement and root-cause analysis.

All work uses a single frozen train/validation split (10,054 / 2,514 images, seed 42) so that
every model, in both problems, is measured on exactly the same held-out images. The RLE codec was
verified by round-trip on 20 real annotations and by visual inspection of mask overlays before any
model was trained.

---

## 2. Exploratory Data Analysis

The EDA established the schema, the class balance, and three facts that directly shaped the
modelling choices. Figures: `outputs/figs/eda_structured.png`, `eda_intensity.png`,
`eda_samples.png`, `eda_outliers.png`; schema table `outputs/figs/schema_table.csv`.

**Record counts, missing values, schema.** 12,568 labelled images; `train.csv` has no missing
values in the fields used; the schema table records dtype, cardinality and an example per column.

**Univariate structure.** Per-class image counts, a defect / defect-free split, and grayscale
intensity histograms (global and per class).

**Bivariate structure.** Defect area by class (violin, log-scaled) and a class co-occurrence
heatmap.

Five conclusions carried into the modelling:

1. **The binary has-defect task is nearly balanced** — 6,666 defective against 5,902 clean
   (53.0% defective) — so the binary problem needs no resampling.
2. **The severe imbalance is between defect classes:** class 3 has 5,150 images against class 2's
   247, a 21:1 ratio. Any reweighting belongs on the multi-class problem, not the binary one.
3. **6.41% of defect images (427 of 6,666) carry more than one class,** so classification must be
   multi-label (four sigmoids + BCE), not four-way softmax. The heatmap shows the concrete pairs:
   classes 3 and 4 co-occur 284 times, classes 1 and 3 co-occur 93 times.
4. **Class 2 is both rare and small in area,** so it is expected to be the hardest to segment, and
   per-class Dice should be split by defect area rather than pooled.
5. **Images are uniformly 1600×256 strips** (a 6.25:1 aspect ratio), so a square-crop pipeline
   would discard most of each frame; the pipeline resizes along the width instead.

Conclusion 3 is the load-bearing one: it fixes the classification head as multi-label before any
model is built.

---

## 3. Problem 1 — Defect Classification

### 3.1 Setup

The head is four independent sigmoids trained with binary cross-entropy, chosen because 6.41% of
defect images carry more than one class (EDA conclusion 3). Grayscale images are replicated to
three channels to use ImageNet weights, resized to 256×800 to preserve the wide aspect ratio, and
the classification threshold is fixed at 0.5 with no per-class tuning on the validation set.
Champion and challenger share an identical optimiser, schedule, augmentation and batch size so the
comparison reflects the architectures.

### 3.2 Choice of loss — imbalance ablation

Three strategies for the 21:1 class imbalance were compared on the champion, measured by
validation macro-F1:

| loss | macro-F1 | class-2 F1 |
|---|---|---|
| BCE | **0.898** | 0.857 |
| focal | 0.886 | 0.805 |
| class positive-weighting | 0.779 | 0.571 |

Plain BCE won. Positive-weighting, the intuitive fix for imbalance, was the worst: a large weight
on class 2 (247 images) destabilised training and actually lowered its F1 to 0.571. `WeightedRandomSampler`
was rejected on principle rather than tested to convergence, because it weights whole images and
427 images here carry several classes at once, so no per-image weight can rebalance all four
classes independently. Loss-level BCE acts per label and does not have that problem.

### 3.3 Results

Both models are strong; the challenger is stronger on every metric.

| metric | ResNet-50 (champion) | EfficientNet-B2 (challenger) |
|---|---|---|
| macro-F1 | 0.9037 | **0.9341** |
| micro-F1 | 0.9146 | **0.9440** |
| subset exact-match accuracy | 0.9181 | **0.9443** |
| macro ROC-AUC | 0.9906 | **0.9959** |
| macro PR-AUC (average precision) | 0.9522 | **0.9792** |

Per-class F1 (validation support in parentheses):

| class (support) | ResNet-50 | EfficientNet-B2 |
|---|---|---|
| 1 (179) | 0.879 | 0.899 |
| 2 (47) | 0.874 | 0.920 |
| 3 (1034) | 0.919 | 0.949 |
| 4 (157) | 0.944 | 0.968 |

EfficientNet-B2's largest gain is on class 2, the rare class — the opposite of what raw capacity
would predict, and consistent with its ~9M parameters being better matched to a 12.5k-image
dataset than ResNet-50's 25.6M. A derived binary has-defect head (ResNet-50) reaches F1 0.972 /
ROC-AUC 0.995, confirming that the difficulty is entirely in the which-class judgement, not in
detecting a defect at all.

### 3.4 Reading the metrics honestly

- **PR-AUC over ROC-AUC.** Class 2 is 47 positives in 2,514 images. A model that finds none of
  them still posts a high ROC-AUC because the enormous negative pool keeps the false-positive rate
  low. Average precision compares against the low base rate and is the honest headline under this
  imbalance — which is why the PR-AUC gap between the models (0.952 vs 0.979) is wider than the
  ROC-AUC gap.
- **No single 4×4 confusion matrix.** The head is multi-label, so an image can appear in two rows
  at once. Four per-class 2×2 matrices are the exact view; a 5×5 dominant-class matrix is readable
  but exact only for the 93.6% of defect images that carry a single class.
- **Threshold fixed at 0.5.** No per-class threshold tuning was done on the reported split, because
  tuning on the same data that is being reported would inflate the numbers.

### 3.5 Architecture trade-offs for this data

ResNet-50's early large receptive field and stable fine-tuning suit the "is a defect present
somewhere on this 1600-pixel strip" judgement, and its wide channels at coarse resolution match
class 3's large diffuse texture, where most of the macro-F1 mass sits. Its costs here are
over-parameterisation for 12.5k images and a 32× downsample that can wash out the thin class-2
line before the head sees it. EfficientNet-B2's squeeze-and-excitation gating is well matched to a
global "defect somewhere" decision and its compound-scaling native resolution (260 px) makes
256×800 less out-of-distribution, but its depthwise-separable convolutions are
memory-bandwidth-bound rather than faster in wall-clock on the T4 (0.216 vs 0.229 s/step), so its
advantage is footprint and accuracy, not throughput. Neither model resolves the thin-defect
problem at 32× downsampling; that is what the segmentation head addresses.

Figures: `outputs/figs/cls_confusion_resnet50.png`, `cls_confusion_effnetb2.png`,
`cls_roc_compare.png`, `cls_pr_compare.png`, `cls_training_curves.png`.

---

## 4. Problem 2 — Defect Segmentation

### 4.1 Setup and the loss bug that had to be fixed first

Both segmentation models use the same ResNet-34 ImageNet encoder — U-Net (champion) and DeepLabV3+
(challenger) — so the comparison is about the decoder. Loss is half BCE, half soft Dice, at native
resolution, batch 8, AdamW with cosine annealing, 15 epochs, seed 42.

The first U-Net run collapsed: training loss fell smoothly while validation Dice went 0.58 → 0.39
→ 0.09 → 0.09 and the false-positive rate fell toward zero — the model had learned to output empty
masks. The cause was in the Dice term, which was averaged over all (image, class) pairs including
empty-ground-truth ones. For an empty pair the score is `smooth / (pred.sum() + smooth)`, maximised
by predicting nothing, and roughly 85–90% of channel-slots here are empty (four classes, usually
at most one present, 47% of images clean), so that term was overwhelmingly an instruction to
predict nothing — the exact failure a Dice loss is meant to prevent. The fix restricts the Dice
term to non-empty-GT pairs and lets BCE supervise the empty ones; three regression tests fail
against the old implementation and pass against the new. A resume checkpoint from the collapsed run
had to be deleted deliberately, because otherwise the corrected run would silently continue the
broken one.

Removing the empty-pair pressure also removed the only thing suppressing over-prediction, so the
false-positive rate rises. That cost is real and is why the results below report a sensitivity
curve rather than a single tuned operating point.

### 4.2 Results

Headline metrics are computed on the 1,417 defect-bearing (image, class) pairs only; behaviour on
defect-free images is reported separately. Both models converged (epoch 15 ties the best epoch).

| metric (defect-only unless noted) | U-Net (champion) | DeepLabV3+ (challenger) |
|---|---|---|
| Dice | 0.7054 | **0.7245** |
| mIoU | 0.5770 | **0.5970** |
| FP rate on defect-free images | 0.3814 | **0.3619** |
| wrong-class FP inside defect images | 0.3771 | **0.2353** |
| all-pairs Dice (Kaggle convention) | 0.7386 | 0.8050 |

Per-class Dice and by defect size (median-area split at 10,203 px):

| breakdown | U-Net | DeepLabV3+ |
|---|---|---|
| class 1 / 2 / 3 / 4 | 0.642 / 0.636 / 0.714 / 0.738 | 0.665 / 0.654 / 0.733 / 0.757 |
| small defects (709 pairs) | 0.681 (14 missed) | 0.696 (9 missed) |
| large defects (708 pairs) | 0.730 (6 missed) | 0.754 (2 missed) |

DeepLabV3+ wins on every axis — Dice, mIoU, false positives, all four classes, both size groups,
fewer defects missed entirely — while using fewer parameters (22.4M vs 24.4M). Its atrous-pooling
decoder's wider receptive field suits the long, thin defects on a 1600×256 strip better than
U-Net's symmetric skip decoder. Both models find defect *size* a harder axis than defect *class*:
the small/large gap (~0.06) exceeds the spread across classes. This refines the EDA prediction —
class 2 was expected to be hardest as the rarest and smallest, and it is joint-weakest, but the
sharper driver turned out to be size rather than class rarity.

### 4.3 The naive metric ranks the do-nothing model higher

Under the Kaggle all-pairs convention, where every empty-versus-empty pair scores 1.0:

- do-nothing baseline: **0.8591**
- U-Net: 0.7386
- DeepLabV3+: 0.8050

A model that predicts nothing beats both trained segmenters on the competition's own metric,
because empty pairs dominate and every false positive forfeits a point that emptiness would have
banked. This is not an argument about the loss; it is a measurement on our validation set that the
naive metric prefers the useless model, and it is exactly why the headline excludes empty ground
truth and reports false positives separately.

### 4.4 Operating-point sensitivity

Because the fixed-loss model over-predicts, the operating point is where the Dice / false-positive
trade lives, so a 30-point grid (6 probability thresholds × 5 minimum-area values) was swept per
model. Applying the rule "lowest false-positive rate among points keeping ≥95% of headline Dice":

| model | headline (0.5 / 300px) | recommended (0.8 / 1000px) |
|---|---|---|
| U-Net | Dice 0.705, FP 0.381 | Dice 0.694, FP 0.238 |
| DeepLabV3+ | Dice 0.725, FP 0.362 | Dice 0.709, FP **0.199** |

Moving the operating point nearly halves the false-positive rate for under 3% of Dice. In an
inspection setting the right point depends on the cost of re-inspecting a clean sheet versus
missing a defect, so the honest deliverable is the curve, not a single tuned number. Figures:
`outputs/figs/seg_compare_01..14.png` (four-panel image / GT / U-Net / DeepLabV3+),
`seg_operating_point_sensitivity.png`, `seg_training_curves.png`.

---

## 5. Model Operations

The full deployment and maintenance plan is in `report/04_model_ops.md`. In summary: the two-stage
inference path runs the classifier as a cheap triage gate and the segmenter only on flagged images,
which keeps throughput high because roughly half the sheets are clean. The maintenance plan
addresses class-distribution drift (the deployed steel mix will differ from the training set),
periodic re-labelling of the low-confidence and false-positive queue that the confidence gate
produces, and a scheduled retrain trigger tied to a monitored drop in the defect-only Dice on a
held-out audit set rather than to a fixed calendar. The operating point is exposed as a
configuration parameter so the plant can move along the sensitivity curve of Section 4.4 without
retraining.

---

## 6. Conclusion

From one dataset we built and evaluated two distinct problems with champion/challenger pairs on
controlled recipes. In both problems the lighter challenger won — EfficientNet-B2 over ResNet-50 in
classification (macro-F1 0.934 vs 0.904), DeepLabV3+ over U-Net in segmentation (defect-only Dice
0.725 vs 0.705 and better on every sub-metric) — a consistent signal that model capacity was not
the binding constraint on a 12.5k-image dataset and that architecture fit mattered more than size.

The findings that generalise beyond the leaderboard:

- **The label distribution should choose the loss, and the intuitive choice was wrong.** BCE beat
  both focal loss and class positive-weighting; the positive-weighting that "should" fix a 21:1
  imbalance instead halved the rare class's F1.
- **Defect size, not class rarity, is the hard axis for localisation.** Per-class numbers hide this;
  the area split exposes it, and it was visible in the EDA before any model existed.
- **A naive averaged metric can rank a useless model first.** On this data a do-nothing segmenter
  wins the all-pairs Dice, so the reported metric must separate defect-only quality from
  false-positive behaviour, and the operating point belongs on a sensitivity curve rather than
  being tuned on the validation set.

For a reader continuing this work: the highest-value next steps are a small-defect-focused loss or
tiling scheme to lift the small-object Dice, a learned confidence gate to shrink the
false-positive queue, and calibration of the classification thresholds against a real
re-inspection cost model rather than a fixed 0.5.

---

## References

1. Severstal: Steel Defect Detection. Kaggle competition dataset.
   https://www.kaggle.com/c/severstal-steel-defect-detection
2. He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition (ResNet). CVPR.
3. Tan, M., Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.
4. Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical
   Image Segmentation. MICCAI.
5. Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., Adam, H. (2018). Encoder-Decoder with Atrous
   Separable Convolution for Semantic Image Segmentation (DeepLabV3+). ECCV.
6. Yakubovskiy, P. Segmentation Models PyTorch. https://github.com/qubvel/segmentation_models.pytorch
7. torchvision model zoo (ResNet-50, EfficientNet-B2 ImageNet weights). PyTorch.

_Reused code is cited above; all model implementations use torchvision and segmentation-models-pytorch
with the standard ImageNet-pretrained backbones. All training and evaluation code is in this
repository under `src/` and `notebooks/`._
