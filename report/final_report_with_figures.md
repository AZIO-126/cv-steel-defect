# Steel Surface Defect Detection: Classification and Segmentation on the Severstal Dataset

**Authors:** Haobo Yang, Anqi Yang, Yue Wang

**Course:** ADSP 32023 Advanced Computer Vision, Final Project

---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction and Dataset](#1-introduction-and-dataset)
3. [Exploratory Data Analysis](#2-exploratory-data-analysis)
4. [Problem 1: Defect Classification](#3-problem-1-defect-classification)
5. [Problem 2: Defect Segmentation](#4-problem-2-defect-segmentation)
6. [Model Operations](#5-model-operations)
7. [Conclusion](#6-conclusion)
8. [References](#references)

---

## Abstract

We study automated inspection of hot-rolled steel surfaces using the Severstal dataset of 12,568
labelled 1600×256 grayscale images across four defect classes. From one dataset we pose two
distinct cognitive problems: multi-label classification (which of four defect types is present)
and pixel-level segmentation (where each defect is). For each problem we train a champion and a
challenger under the same recipe, each early-stopped on its validation metric, so the comparison
isolates the model rather than the training setup. In classification, EfficientNet-B2 (challenger)
reaches a macro-F1 of 0.934 against ResNet-50's 0.904; a loss-function ablation shows plain BCE
beats focal loss and class positive-weighting for this label distribution. In segmentation,
DeepLabV3+ (challenger) reaches a defect-only Dice of 0.725 against U-Net's 0.705 on a shared
ResNet-34 encoder, and wins on every sub-metric we report. The segmentation work also surfaces a
measurement trap specific to this dataset: under the competition's own all-pairs Dice convention a
model that predicts nothing scores 0.859, higher than either trained model, because
empty-versus-empty pairs are scored as perfect. We therefore report defect-only metrics as the
headline, false-positive behaviour separately, and an operating-point sensitivity curve, because
the right precision and recall trade for an inspection line is a business decision rather than a
number to tune on the validation set.

---

## 1. Introduction and Dataset

Surface-defect inspection is a natural computer-vision problem: the defects are visible, the
labels are physical, and both "is there a defect" and "where is it" matter to a production line.
The Severstal dataset provides 12,568 labelled grayscale images, each a 1600×256 strip of rolled
steel, annotated with run-length-encoded masks for four defect classes.

The dataset supports two genuinely different cognitive problems, which is what the project calls
for:

- **Classification.** A per-image, multi-label judgement of which of the four defect classes are
  present. This is the triage question on an inspection line.
- **Segmentation.** A per-pixel mask of each defect. This is the localisation question that drives
  measurement and root-cause analysis.

All work uses a single frozen train/validation split (10,054 / 2,514 images, seed 42) so that
every model, in both problems, is measured on exactly the same held-out images. The RLE codec was
verified by round-trip on 20 real annotations and by visual inspection of mask overlays before any
model was trained.

---

## 2. Exploratory Data Analysis

The EDA established the schema, the class balance, and three facts that directly shaped the
modelling choices. Figures: `outputs/figs/eda_structured.png`, `eda_intensity.png`,
`eda_samples.png`, `eda_outliers.png`; schema table `outputs/figs/schema_table.csv`.


![](/Users/yanghaobo/Projects/cv-steel-defect/outputs/figs/eda_structured.png)

![](/Users/yanghaobo/Projects/cv-steel-defect/outputs/figs/eda_intensity.png)

![](/Users/yanghaobo/Projects/cv-steel-defect/outputs/figs/eda_samples.png)

![](/Users/yanghaobo/Projects/cv-steel-defect/outputs/figs/eda_outliers.png)


**Record counts, missing values, schema.** 12,568 labelled images; `train.csv` has no missing
values in the fields used; the schema table records dtype, cardinality and an example per column.

**Univariate structure.** Per-class image counts, a defect / defect-free split, and grayscale
intensity histograms (global and per class).

**Bivariate structure.** Defect area by class (violin, log-scaled) and a class co-occurrence
heatmap.

Five conclusions carried into the modelling:

1. **The binary has-defect task is nearly balanced.** 6,666 defective against 5,902 clean
   (53.0% defective), so the binary problem needs no resampling.
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

## 3. Problem 1: Defect Classification

### 3.1 Setup

The head is four independent sigmoids trained with binary cross-entropy, chosen because 6.41% of
defect images carry more than one class (EDA conclusion 3). Grayscale images are replicated to
three channels to use ImageNet weights, resized to 256×800 to preserve the wide aspect ratio, and
the classification threshold is fixed at 0.5 with no per-class tuning on the validation set.
Champion and challenger share the same optimiser, schedule, augmentation and batch size, and each
is early-stopped on validation macro-F1, so the comparison reflects the architectures. Early
stopping ended ResNet-50 at epoch 8 (best at epoch 5) and EfficientNet-B2 at epoch 12 (best at
epoch 11).

### 3.2 Choice of loss: imbalance ablation

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
classes independently. Loss-level BCE acts per label and does not have that problem. The figures in
this table are the best validation macro-F1 reached during each sweep; the 0.904 headline in
Section 3.3 is the final evaluation of the reloaded BCE checkpoint, which is why it is slightly
higher than the 0.898 here.

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

EfficientNet-B2's largest gain is on class 2, the rare class. This is the opposite of what raw
capacity would predict, and it is consistent with its roughly 9M parameters being better matched
to a 12.5k-image dataset than ResNet-50's 25.6M. A separately trained binary has-defect head
(ResNet-50) reaches F1 0.972 and ROC-AUC 0.995, which confirms that the difficulty is in the
which-class judgement, not in detecting a defect at all.

### 3.4 Reading the metrics honestly

- **PR-AUC over ROC-AUC.** Class 2 is 47 positives in 2,514 images. A model that finds none of
  them still posts a high ROC-AUC because the enormous negative pool keeps the false-positive rate
  low. Average precision compares against the low base rate and is the honest headline under this
  imbalance, which is why the PR-AUC gap between the models (0.952 vs 0.979) is wider than the
  ROC-AUC gap.
- **No single 4×4 confusion matrix.** The head is multi-label, so an image can appear in two rows
  at once. Four per-class 2×2 matrices are the exact view; a 5×5 dominant-class matrix is readable
  but exact only for the 93.6% of defect images that carry a single class.
- **Threshold fixed at 0.5.** No per-class threshold tuning was done on the reported split, because
  tuning on the same data that is being reported would inflate the numbers.

### 3.5 Architecture trade-offs for this data

Both backbones are judged against the same data: 12,568 grayscale strips, a 21:1 class imbalance,
6.41% multi-label images, and defects ranging from a large diffuse patch to a line a few pixels
wide.

ResNet-50 (champion), pros:

- Its early large receptive field suits the judgement of whether a defect sits anywhere along a
  1600-pixel strip, without needing the whole frame at full resolution.
- Residual blocks fine-tune stably at 256×800 with no warmup or gradient clipping, which lets the
  same recipe be shared with the challenger so the comparison stays fair.
- Wide channels at coarse resolution match class 3, the large diffuse texture that holds most of
  the macro-F1 mass.

ResNet-50, cons:

- At 25.6M parameters it is over-parameterised for 12.5k images, so class 2's 247 images cannot
  constrain it and its class-2 behaviour is driven by the loss more than by the data.
- Its 32× downsample turns a 256×800 input into an 8×25 map, which can wash out the thin class-2
  line before the head sees it. This is the concrete reason class-2 F1 is the weakest.
- It was the slower model per epoch at this resolution, a real cost when one GPU serves both the
  classification and the segmentation phase.

EfficientNet-B2 (challenger), pros:

- At about 9M parameters it fits a 12.5k-image dataset far better, which is where the rare classes
  benefit most and where its class-2 gain comes from.
- Squeeze-and-excitation gating uses global context, which matches the "is there a defect
  somewhere in this frame" decision this strip geometry poses.
- Its compound-scaling native resolution of 260px makes a 256×800 input less out-of-distribution
  than it is for a backbone trained at 224.

EfficientNet-B2, cons:

- Its parameter efficiency does not convert into speed here. Depthwise-separable convolutions are
  memory-bandwidth-bound on the T4, so it ran only marginally faster per epoch than ResNet-50
  despite carrying about a third of the parameters. The gain is footprint and accuracy, not
  throughput.
- Its BatchNorm statistics depend on batch composition, and with class 2 under 2% prevalence many
  batches carry no class-2 example, which makes its validation curve less stable.
- It shares ResNet-50's 32× downsample and has a narrower stem, so it discards the thin class-2
  line at least as early. Neither model solves the small-defect problem, which is what the
  segmentation head addresses.

Figures: `outputs/figs/cls_confusion_resnet50.png`, `cls_confusion_effnetb2.png`,
`cls_roc_compare.png`, `cls_pr_compare.png`, `cls_training_curves.png`.


![](/Users/yanghaobo/Projects/cv-steel-defect/outputs/figs/cls_training_curves.png)

![](/Users/yanghaobo/Projects/cv-steel-defect/outputs/figs/cls_confusion_resnet50.png)

![](/Users/yanghaobo/Projects/cv-steel-defect/outputs/figs/cls_confusion_effnetb2.png)

![](/Users/yanghaobo/Projects/cv-steel-defect/outputs/figs/cls_roc_compare.png)

![](/Users/yanghaobo/Projects/cv-steel-defect/outputs/figs/cls_pr_compare.png)


---

## 4. Problem 2: Defect Segmentation

### 4.1 Setup and loss design

Both segmentation models use the same ResNet-34 ImageNet encoder, U-Net (champion) and DeepLabV3+
(challenger), so the comparison is about the decoder. Loss is half BCE and half soft Dice, at
native resolution, batch 8, AdamW with cosine annealing, 15 epochs, seed 42.

The Dice term is computed only on (image, class) pairs with a non-empty ground truth, and BCE
supervises the empty pairs. This design decision matters for this dataset. Averaging soft Dice over
empty-ground-truth pairs as well gives, for each empty pair, a score of `smooth / (pred.sum() +
smooth)`, which is maximised by predicting nothing. Roughly 85 to 90% of channel-slots here are
empty (four classes, usually at most one present, and 47% of images clean), so an all-pairs Dice
term becomes an overwhelming instruction to predict empty masks, the exact failure a Dice loss is
meant to prevent. Restricting Dice to non-empty pairs and letting BCE punish false positives on the
empty ones removes that pressure. Three regression tests hold the behaviour in place: they fail on
the all-pairs formulation and pass on the restricted one.

Restricting the Dice term also removes the main force suppressing over-prediction, so the
false-positive rate rises. That cost is real, and it is why the results below report a sensitivity
curve rather than a single tuned operating point.

### 4.2 Results

Headline metrics are computed on the 1,417 defect-bearing (image, class) pairs only; behaviour on
defect-free images is reported separately. Both models converged, with the last epoch within 0.001
Dice of the best (U-Net best at epoch 14, DeepLabV3+ at epoch 15).

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

DeepLabV3+ wins on every axis (Dice, mIoU, false positives, all four classes, both size groups,
and fewer defects missed entirely) while using fewer parameters (22.4M vs 24.4M). Its atrous
pooling gives the decoder a wider receptive field, which suits the long thin defects on a 1600×256
strip better than U-Net's symmetric skip decoder.

Two error axes stand out, and defect class separates performance more than defect size does. For
both models classes 1 and 2 score about 0.64 while classes 3 and 4 score 0.73 to 0.76, a spread of
about 0.10; the small-versus-large gap is about 0.05, roughly half as large. This matches the EDA
prediction that class 2, the rarest and smallest, would be among the hardest, though class 1 is
just as weak, so the pattern is about defect type rather than rarity alone. Small defects are
genuinely harder than large ones as well, but by less than class identity.

U-Net (champion) is a strong baseline: its symmetric skip connections carry fine spatial detail to
the decoder, it trains stably at native 1600×256 with the corrected loss, and it needs no atrous
tuning. Its costs on this data are a higher false-positive rate (0.38 vs 0.36), a higher
wrong-class rate inside defect images (0.38 vs 0.24), and more small defects missed entirely (14 vs
9). DeepLabV3+ (challenger) pros: the atrous spatial pyramid captures the wide context a long strip
needs, it reaches higher Dice on every class, and it does so with fewer parameters. Its cons: its
output stride can blur the sharpest mask boundaries, its false-positive rate is still high in
absolute terms (0.36), and like U-Net it depends on an operating point that has to be chosen off
the sensitivity curve rather than read off validation.

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

Moving the operating point cuts the false-positive rate by a third to a half (U-Net 0.38 to 0.24,
DeepLabV3+ 0.36 to 0.20) for under 3% of Dice. In an inspection setting the right point depends on
the cost of re-inspecting a clean sheet versus missing a defect, so the honest deliverable is the
curve, not a single tuned number. Figures: `outputs/figs/seg_compare_01..14.png` (four panels,
image, ground truth, U-Net, DeepLabV3+), `seg_operating_point_sensitivity.png`,
`seg_training_curves.png`.


![](/Users/yanghaobo/Projects/cv-steel-defect/outputs/figs/seg_training_curves.png)

![](/Users/yanghaobo/Projects/cv-steel-defect/outputs/figs/seg_compare_03_354760e3e.png)

![](/Users/yanghaobo/Projects/cv-steel-defect/outputs/figs/seg_compare_08_6181b0a92.png)

![](/Users/yanghaobo/Projects/cv-steel-defect/outputs/figs/seg_operating_point_sensitivity.png)


---

## 5. Model Operations

The full deployment and maintenance plan is in `report/04_model_ops.md`. In summary, the two-stage
inference path runs the classifier as a cheap triage gate and the segmenter only on flagged images,
which keeps throughput high because roughly half the sheets are clean. The deployed classifier is
EfficientNet-B2, the challenger that won every classification metric and is also the lighter model
to serve; the segmenter is DeepLabV3+, the winning challenger there. The maintenance plan addresses
class-distribution drift (the deployed steel mix will differ from the training set), periodic
re-labelling of the low-confidence and false-positive queue that the confidence gate produces, and
a retrain trigger tied to a monitored drop in the defect-only Dice on a held-out audit set rather
than to a fixed calendar. The operating point is exposed as a configuration parameter so the plant
can move along the sensitivity curve of Section 4.4 without retraining.

---



![](/Users/yanghaobo/Projects/cv-steel-defect/outputs/figs/deploy_architecture.png)


## 6. Conclusion

From one dataset we built and evaluated two distinct problems with champion and challenger pairs on
controlled recipes. In both problems the lighter challenger won: EfficientNet-B2 over ResNet-50 in
classification (macro-F1 0.934 vs 0.904), and DeepLabV3+ over U-Net in segmentation (defect-only
Dice 0.725 vs 0.705, and better on every sub-metric). This is a consistent signal that model
capacity was not the binding constraint on a 12.5k-image dataset, and that architecture fit
mattered more than size.

The findings that generalise beyond the leaderboard:

- The label distribution should choose the loss, and the intuitive choice was wrong. BCE beat both
  focal loss and class positive-weighting; the positive-weighting that should fix a 21:1 imbalance
  instead halved the rare class's F1.
- Defect class separates segmentation quality more than defect size does. Classes 1 and 2 trail
  classes 3 and 4 by about 0.10 in Dice, while the small-versus-large gap is about half that. Both
  axes were visible in the EDA before any model existed.
- A naive averaged metric can rank a useless model first. On this data a do-nothing segmenter wins
  the all-pairs Dice, so the reported metric must separate defect-only quality from false-positive
  behaviour, and the operating point belongs on a sensitivity curve rather than being tuned on the
  validation set.

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
