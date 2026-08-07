# Phase 4 — Segmentation results

Problem 2 is pixel-level defect segmentation on the Severstal steel surface dataset: given a
1600×256 grayscale image, predict a mask for each of the four defect classes. We trained two
architectures on the same ResNet-34 ImageNet encoder so the comparison is about the decoder, not
the backbone: U-Net as the champion and DeepLabV3+ as the challenger. Both use a combined loss
(half BCE, half soft Dice), batch 8 at native resolution, AdamW with cosine annealing, 15 epochs,
seed 42, on the frozen 10,054 / 2,514 train/val split.

## The loss bug that had to be fixed first

The first U-Net run collapsed. Training loss fell smoothly from 0.51 to 0.13 while validation
Dice went 0.58 → 0.39 → 0.09 → 0.09 and the false-positive rate fell toward zero — the model had
learned to output empty masks, and the "best" checkpoint was epoch 1, an almost untrained
snapshot. The cause was in the Dice term: it was averaged over all (image, class) pairs including
the ones with an empty ground truth. For an empty pair the intersection is zero, so the score is
`smooth / (pred.sum() + smooth)`, which is maximised by predicting nothing. About 85–90% of the
channel-slots here are empty (four classes, usually at most one present per image, and 47% of
images carry no defect at all), so that term was overwhelmingly an instruction to predict
nothing — the exact failure a Dice loss is supposed to prevent.

The fix restricts the Dice term to pairs with a non-empty ground truth and lets BCE supervise the
empty pairs, which is the right tool for "nothing here" and already penalises false positives.
Measured directly on the two implementations, the old version prefers an all-empty prediction by
0.59 on an all-empty batch and dilutes a real signal by 0.09 when one defect pair sits among seven
empty ones; the new version is indifferent in both cases (three regression tests, all passing).
Both models below use the fixed loss.

Honest note on the trade the fix makes: removing the empty-pair pressure from the Dice term also
removed the only thing holding down over-prediction, so the false-positive rate rises. That cost
is real and is why we report a sensitivity curve rather than a single tuned number.

## Champion vs challenger

All headline figures are computed on the 1,417 defect-bearing (image, class) pairs only. Pairs
with an empty ground truth are excluded from the headline because scoring empty-vs-empty as a
perfect 1.0 lets a do-nothing model win; behaviour on defect-free images is reported separately as
a false-positive rate. Both models converged (epoch 15 essentially ties the best epoch), so this
is a settled comparison, not a truncated one.

| metric (defect-only unless noted) | U-Net (champion) | DeepLabV3+ (challenger) |
|---|---|---|
| Dice | 0.7054 | **0.7245** |
| mIoU | 0.5770 | **0.5970** |
| FP rate on defect-free images | 0.3814 (450 / 1180) | **0.3619 (427 / 1180)** |
| wrong-class FP inside defect images | 0.3771 | **0.2353** |
| all-pairs Dice (Kaggle convention) | 0.7386 | 0.8050 |
| best epoch | 14 | 15 |

Per-class Dice:

| class (n val pairs) | U-Net | DeepLabV3+ |
|---|---|---|
| 1 (179) | 0.6422 | 0.6654 |
| 2 (47) | 0.6358 | 0.6538 |
| 3 (1034) | 0.7145 | 0.7330 |
| 4 (157) | 0.7383 | 0.7575 |

By defect size (split at the median GT area, 10,203 px):

| group | U-Net Dice (missed) | DeepLabV3+ Dice (missed) |
|---|---|---|
| small (709 pairs, median 4,056 px) | 0.6807 (14 missed) | 0.6955 (9 missed) |
| large (708 pairs, median 25,957 px) | 0.7300 (6 missed) | 0.7536 (2 missed) |

DeepLabV3+ wins on every axis: higher Dice and mIoU, a lower false-positive rate, better on all
four classes, better on both size groups, fewer defects missed entirely, and a much lower
wrong-class false-positive rate. It also has fewer trainable parameters (22.4M vs 24.4M) and was
faster per step in preflight. So the challenger is the stronger model here — the same outcome as
the classification problem, where the lighter EfficientNet-B2 challenger beat the ResNet-50
champion. The atrous-pooling decoder's wider receptive field appears to suit these long, thin
defects on a 1600×256 strip better than U-Net's symmetric skip decoder.

Both models find defect size a harder axis than defect class: the small/large gap (~0.06) is wider
than the spread across classes. Phase 2 predicted class 2 would be hardest because it is the rarest
(247 images) and smallest — directionally right on difficulty, but class 2 (0.65) lands level with
class 1 rather than far below everything, so the sharper driver is size, not class rarity. Worth
stating plainly rather than claiming Phase 2 called it exactly.

## The naive metric ranks the do-nothing model higher

Under the Kaggle all-pairs convention, where every empty-vs-empty pair scores a perfect 1.0:

- do-nothing baseline: **0.8591**
- U-Net: 0.7386
- DeepLabV3+: 0.8050

A model that predicts nothing beats both working segmenters on the competition's own metric,
because empty pairs dominate and every false positive forfeits a point that emptiness would have
banked. This is not an argument about which loss to use — it is a measurement, on our own
validation set, that the naive metric prefers the useless model, which is exactly why our headline
excludes empty ground truth and reports false positives separately. DeepLabV3+ sits closer to the
do-nothing number than U-Net does precisely because its lower false-positive rate wastes fewer of
those empty-vs-empty points.

## Operating-point sensitivity

The headline uses a fixed, pre-registered operating point (probability threshold 0.5, minimum
class area 300 px) with no tuning on the validation set. Because the fixed-loss model over-predicts,
the operating point is where the Dice-vs-false-positive trade lives, so we swept a 30-point grid
(6 thresholds × 5 minimum-area values) per model rather than picking one flattering number.

Applying the rule "lowest false-positive rate among points that keep at least 95% of the headline
Dice":

| model | headline (thr 0.5 / 300px) | recommended (thr 0.8 / 1000px) |
|---|---|---|
| U-Net | Dice 0.7054, FP 0.3814 | Dice 0.6941, FP 0.2381 |
| DeepLabV3+ | Dice 0.7245, FP 0.3619 | Dice 0.7095, FP **0.1992** |

Moving the operating point nearly halves the false-positive rate for under 3% of Dice. In a
defect-inspection setting that choice is a business decision — how much re-inspection cost a missed
defect is worth — so the honest deliverable is the curve, not a single tuned point. See
`figs/seg_operating_point_sensitivity.png`.

## Artifacts

- `outputs/metrics/seg_unet.json`, `seg_deeplabv3p.json` — full metrics per model (headline,
  per-class, per-area, false positives, all-pairs reference, empty-prediction baseline, full history).
- `outputs/metrics/seg_sensitivity.json` — 30-point operating-point grid per model + recommended point.
- `outputs/figs/seg_compare_01..14.png` — 14 four-panel qualitative comparisons (image, ground
  truth, U-Net, DeepLabV3+).
- `outputs/figs/seg_operating_point_sensitivity.png`, `seg_training_curves.png`.
- `outputs/ckpt/` — best-epoch weights for both models. Optimizer/resume state (`*_last.pth`,
  ~563 MB) is deliberately excluded from the deliverable.

## Process notes carried into Phase 6

- A resume/checkpoint mechanism is a hazard after a code change, not just insurance: the stale
  `*_last.pth` would have let the fixed run silently continue the collapsed one. Delete
  checkpoints deliberately when the thing that produced them changed.
- A busy Colab kernel cannot be queried or written to; any live progress view must be parsed from
  the running cell's stdout, and any file push waits for the kernel to go idle.
- Index off the live cell map, never a local copy's positions — the notebook's cell indices had
  drifted from the local file, and driving by the stale positions would have run the wrong cells
  and produced believable output with the sensitivity analysis simply absent.
