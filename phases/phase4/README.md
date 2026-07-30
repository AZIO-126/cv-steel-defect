# Phase 4 — Problem 2: defect segmentation

**Owner** Segmentation+Ops · **Day 3–10** · **Worth ~20 marks**
**Depends on** phase 0 (split), phase 1 (verified RLE decode)

## Grading criteria (verbatim)
> Build at least two competing (champion-challenger) models per problem
> Perform detailed model performance evaluation and comparison with appropriate metrics
> Describe the algorithms implemented and their pros and cons with respect to the data

## Steps
1. **champion: U-Net** via `segmentation-models-pytorch`, ResNet-34 or ResNet-50 encoder.
2. **challenger: DeepLabV3+**, same library and **same encoder family** — otherwise you are
   comparing encoders, not architectures.
3. **Loss: Dice + BCE combined.** Not pure BCE: defect pixels are a tiny fraction of the
   image, so pure BCE drives the model toward predicting all-background. Put this reasoning
   in the report — it shows the loss choice was driven by the data.
4. **Metrics** → `outputs/metrics/seg_{model}.json`: **Dice / mIoU** (matching the Kaggle
   competition's official metric) and per-class Dice.
5. **Required grouped analysis** — split the validation set into small-defect and
   large-defect groups by area and report Dice separately for each. This pays off phase 2's
   area analysis and is the strongest evidence in the report that the data was understood.
6. **The trap that silently inflates the score:** the validation set contains many
   defect-free images with empty masks. Counting them in the headline metric lets a model
   that predicts nothing score highly. So:
   - compute the **headline Dice/mIoU on defect-bearing images only**
   - report a **separate false-positive rate on the defect-free images**
   - state both, and the reason, explicitly in the report
7. **Qualitative comparison**: ≥8 four-panel figures — original / ground-truth mask /
   U-Net prediction / DeepLabV3+ prediction.

## Artifacts
- `notebooks/02b_segmentation.ipynb`
- `outputs/ckpt/unet_*.pth`, `outputs/ckpt/deeplabv3p_*.pth`
- `outputs/metrics/seg_unet.json`, `seg_deeplabv3p.json`
- ≥8 comparison figures in `outputs/figs/`
- `phases/phase4/RESULT.md` — metric table incl. the small/large split and the FP rate

## DONE test
- Both models evaluated on the same split
- Headline metric computed on defect-only images, with the defect-free FP rate reported
  separately — both numbers present in the json
- ≥8 four-panel qualitative figures saved
