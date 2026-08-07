# Presentation Outline — Steel Defect Detection (20 pts, ~20 min, 4 members)

One rubric area per member so every graded section is spoken to. Keep slides visual — one
takeaway each, figures/tables from `outputs/figs/`, minimal text. Author names go on the title
slide. Deliver as pdf or pptx.

---

## Member 1 — Problem & Data & EDA (~5 min, 4 slides)

**Slide 1 — Title.** Project title, all author names, course. One hero image: a steel strip with a
mask overlay (`eda_samples.png` crop).

**Slide 2 — Why this dataset, why two problems.** Severstal: 12,568 grayscale 1600×256 strips, 4
defect classes, RLE masks. From one dataset, two cognitive problems: classification (which defects
— triage) and segmentation (where — localisation). One line on the frozen 10,054/2,514 split.

**Slide 3 — EDA headline (the figure).** Show `eda_structured.png`. Say the three facts that drove
modelling: binary balanced (53% defective, no resampling), but classes 21:1 imbalanced (class 3
5,150 vs class 2 247), and 6.41% multi-label → four sigmoids, not softmax.

**Slide 4 — What the data shape forces.** 1600×256 (6.25:1) → resize along width, no square crop.
Class 2 rare AND small → expect it hardest, report Dice by area. Bridge to Member 2.

## Member 2 — Problem 1 Classification (~5 min, 4 slides)

**Slide 5 — Setup + loss ablation (table).** Multi-label 4-sigmoid + BCE. Show the ablation:
BCE 0.898 > focal 0.886 > pos-weight 0.779. The punchline: pos-weight, the "obvious" imbalance fix,
halved class-2 F1 (0.857→0.571). Let the data pick the loss.

**Slide 6 — Champion vs challenger (table).** ResNet-50 0.9037 vs EfficientNet-B2 0.9341 macro-F1;
challenger wins every metric. Per-class F1 row. The lighter model wins — capacity wasn't the
constraint.

**Slide 7 — ROC & PR (figures).** `cls_roc_compare.png` + `cls_pr_compare.png`. Why PR is the honest
read under imbalance (class 2 = 47 positives). Confusion: `cls_confusion_effnetb2.png`.

**Slide 8 — Why multi-label / no single confusion matrix.** One line each: an image can be two rows
at once; four 2×2 matrices are the exact view. Bridge to segmentation (32× downsample loses thin
defects → need pixel head).

## Member 3 — Problem 2 Segmentation (~5 min, 4–5 slides)

**Slide 9 — Setup + the loss bug.** U-Net vs DeepLabV3+, same ResNet-34 encoder (fair decoder
comparison). The collapse story: averaging Dice over empty-GT pairs rewards predicting nothing
(val Dice 0.58→0.09 while train loss falls). Fix: Dice on non-empty pairs, BCE for empty. This is a
strong "we caught a subtle bug" slide.

**Slide 10 — Champion vs challenger (table + curves).** U-Net 0.7054 vs DeepLabV3+ 0.7245 defect-only
Dice; challenger wins every axis. `seg_training_curves.png`. Per-class + area split (size is the
harder axis than class).

**Slide 11 — Qualitative comparison (the money slide).** 2–3 of `seg_compare_*.png` four-panels
(image / GT / U-Net / DeepLabV3+). Let the masks speak.

**Slide 12 — The metric trap (key insight).** Do-nothing all-pairs Dice 0.8591 beats both trained
models (U-Net 0.7386, DeepLabV3+ 0.8050). Why: empty-vs-empty scores 1.0. So we headline defect-only
+ report FP separately. This is the single most impressive analytical point — give it a full slide.

**Slide 13 — Operating-point sensitivity.** `seg_operating_point_sensitivity.png`. thr 0.8/1000px
nearly halves FP (0.36→0.20) for <3% Dice. The operating point is a business decision → a curve, not
a tuned number.

## Member 4 — Model Ops & Conclusion (~5 min, 3–4 slides)

**Slide 14 — Deployment architecture.** `deploy_architecture.png`. Two-stage: classifier triages,
segmenter runs only on flagged sheets (≈half are clean → throughput). Operating point exposed as
config so the plant slides along Slide 13's curve without retraining.

**Slide 15 — Maintenance & retrain.** Class-distribution drift, low-confidence/FP re-labelling queue,
retrain triggered by a monitored Dice drop on an audit set (not a calendar).

**Slide 16 — Conclusion.** Three generalising findings: (1) the label distribution should choose the
loss and the intuitive choice was wrong; (2) defect SIZE not class rarity is the hard axis; (3) a
naive averaged metric can rank a useless model first. In both problems the lighter challenger won.

**Slide 17 — Next steps + Q&A.** Small-defect loss/tiling, learned confidence gate, threshold
calibration against a real re-inspection cost model. Thank you.

---

## Timing / delivery notes
- 17 slides / 20 min ≈ 70s/slide; the money slides (5 ablation, 9 loss bug, 12 metric trap, 13
  sensitivity) deserve more, the bridges less.
- Every claim on a slide has a figure or table behind it in `outputs/`. No bullet lists of prose.
- Rehearse the two hand-offs (Slide 4→5, Slide 8→9) so the problem transition is clean.
