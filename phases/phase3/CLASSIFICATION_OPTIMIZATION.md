# Classification optimization - where to run it

This folder mirrors files that should be added to the root of
`AZIO-126/cv-steel-defect`. It does not replace the existing notebook or frozen
`splits/train.csv` and `splits/val.csv`.

Use a new branch such as `agent/classification-resolution-calibration`. Run the
training commands in Google Colab with a GPU because the full-width and tiled
experiments are too slow for a normal laptop.

## Files to add to the repository

- `src/make_inner_split.py`: makes inner-train/inner-calibration from outer train only.
- `src/cls_optimization.py`: width tiling, horizontal-flip TTA, threshold calibration, metrics.
- `src/cls_train_optimized.py`: complete EfficientNet-B2/ResNet-50 training entrypoint.
- `tests/test_classification_optimization.py`: fast synthetic-data checks; no Kaggle data needed.

## Colab: run these cells in order

Assume the repository is already available at `/content/cv-steel-defect` and
the Kaggle images are under `data/train_images`.

```bash
%cd /content/cv-steel-defect
!pip install -r requirements.txt
!python -m unittest discover -s tests -p 'test_classification_optimization.py'
```

### 1. Create the legal tuning split

This reads only `splits/train.csv`. `splits/val.csv` is passed only to prove
that no outer-validation image leaks into the inner split.

```bash
!python src/make_inner_split.py \
  --index data/index.csv \
  --outer-train splits/train.csv \
  --outer-val splits/val.csv \
  --out-dir splits \
  --seed 142
```

Commit these generated files so everyone uses the same calibration data:

- `splits/inner_train.csv`
- `splits/inner_calibration.csv`
- `splits/inner_split_manifest.json`

### 2. Run the three resolution candidates

Start with EfficientNet-B2 because it is the current winning classifier.
Reduce batch size if Colab runs out of GPU memory.

```bash
# Existing 256x800 baseline under the new inner protocol
!python src/cls_train_optimized.py \
  --tag effb2_resize800_inner --arch effnetb2 --mode resize_800 \
  --epochs 15 --batch-size 8

# Preserve the complete 256x1600 strip
!python src/cls_train_optimized.py \
  --tag effb2_fullwidth_inner --arch effnetb2 --mode full_width \
  --epochs 15 --batch-size 4

# Two overlapping width tiles; bag-level max pooling keeps image-level labels valid
!python src/cls_train_optimized.py \
  --tag effb2_tiles_inner --arch effnetb2 --mode tiles \
  --tile-width 896 --overlap 192 --epochs 15 --batch-size 2
```

Choose the configuration with the highest `best_inner_macro_f1_at_0.5`. Do not
look at outer-val metrics while choosing.

### 3. Add horizontal-flip TTA to the winner

Example if full width won:

```bash
!python src/cls_train_optimized.py \
  --tag effb2_fullwidth_tta_inner --arch effnetb2 --mode full_width --tta \
  --epochs 15 --batch-size 4
```

TTA is applied during evaluation; the underlying training recipe remains the
same. Keep it only if the inner-calibration metric improves.

### 4. Freeze four class-specific thresholds

Replace the example tag in all three paths if a different configuration wins.

```bash
!python src/cls_optimization.py calibrate \
  --labels outputs/metrics/inner_labels_effb2_fullwidth_tta_inner.npy \
  --probs outputs/metrics/inner_probs_effb2_fullwidth_tta_inner.npy \
  --out outputs/metrics/cls_thresholds_inner.json \
  --split-name inner_calibration
```

The calibrator refuses to run if the split name is `outer_val`.

### 5. One final outer-validation run

Use the winning architecture/mode/TTA, the winning inner epoch count, and the
frozen threshold file. `--final-fit` disables early stopping and fits on the
complete outer train for exactly the requested number of epochs. Outer val is
then evaluated once.

```bash
!python src/cls_train_optimized.py \
  --tag effb2_final \
  --arch effnetb2 --mode full_width --tta \
  --train-split splits/train.csv \
  --eval-split splits/val.csv \
  --epochs 11 --batch-size 4 \
  --thresholds outputs/metrics/cls_thresholds_inner.json \
  --final-fit --allow-outer-val
```

Replace `--mode`, `--tta`, and `--epochs` with the actual inner winner. The
final report metrics are written to
`outputs/metrics/history_effb2_final.json` under `final_outer_metrics`.

## What to compare in the report

For every candidate, keep the same seed, inner split, augmentation, optimiser,
and metric implementation. Report:

- macro/micro F1 and per-class F1;
- the four frozen thresholds;
- class-2 positives in inner calibration and outer val;
- GPU time per epoch and inference time;
- whether full width or tiling changed class-2 recall without damaging class 1.

Do not claim class 2 is the final model's lowest-F1 class unless the new outer
metrics actually show that. In the current EfficientNet-B2 result, class 1 is
lower than class 2.
