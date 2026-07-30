# Phase 3 — Problem 1: defect classification

**Owner** Classification+Lead · **Day 2–10** · **Worth ~20 marks**
**Depends on** phase 0 (split), phase 1 (index.csv), phase 2 (single- vs multi-label answer)

## Grading criteria (verbatim)
> Build at least two competing (champion-challenger) models per problem
> Print detailed model evaluation and comparative metrics (e.g. F1, Accuracy, Confusion Matrix, ROC etc.)
> Describe the algorithms implemented and their pros and cons with respect to the data

## Steps
1. **Task definition** — binary has-defect first as a baseline, then the 4-class problem.
   Head type comes from phase 2's co-occurrence number: multi-label (4 sigmoids + BCE) if
   images commonly carry several classes, else 4-way softmax + cross-entropy.
2. **Grayscale into an ImageNet backbone** — images are 1-channel, pretrained weights are
   3-channel. Two options: replicate the channel 3×, or replace the first conv and average
   its weights across input channels. Try both if time allows and put it in the comparison;
   otherwise use channel replication and **state the reason in the report**.
3. **champion: ResNet-50** (ImageNet pretrained).
4. **challenger: EfficientNet-B2** (ImageNet pretrained). Same input size, same split,
   same augmentation — only the architecture differs, or the comparison proves nothing.
5. **Class imbalance** (there are many defect-free images). Try at least two of:
   class-weighted loss · `WeightedRandomSampler` · focal loss. Report the comparison.
6. **Augmentation** (albumentations): horizontal/vertical flip, brightness/contrast jitter.
   **Avoid strong geometric distortion** (large rotations, elastic transforms) — it
   destroys the shape cues that distinguish defect classes.
7. **Training**: AdamW, cosine schedule, early stopping on val macro-F1, fixed seed.
8. **Metrics** → `outputs/metrics/cls_{model}.json`:
   accuracy · **per-class F1 + macro F1** · confusion matrix · **ROC-AUC and PR curves**.
   Include both ROC and PR, and explain in the report that under this much imbalance the
   PR curve is the more honest view.

## Artifacts
- `notebooks/02a_classification.ipynb`
- `outputs/ckpt/resnet50_*.pth`, `outputs/ckpt/effnetb2_*.pth`
- `outputs/metrics/cls_resnet50.json`, `cls_effnetb2.json`
- confusion matrix + ROC + PR figures in `outputs/figs/`
- `phases/phase3/RESULT.md` — metric table + **3+ pros and 3+ cons per model**

## DONE test
- Both models evaluated on the **same** `splits/val.csv`
- Metric json files exist for both and contain every metric named above
- The pros/cons list has ≥3 entries per model and refers to *this* dataset
  (imbalance, grayscale, small defects), not generic textbook properties
