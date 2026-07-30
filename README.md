# CV Final Project — Severstal Steel Defect Detection

ADSP 32023 Advanced Computer Vision with Deep Learning · due **2026-08-15**

Two cognitive problems from one dataset's native labels:
1. **Classification** — does this steel sheet have a defect, and which of 4 classes
2. **Segmentation** — pixel-level defect contour

Dataset: https://www.kaggle.com/competitions/severstal-steel-defect-detection/data
(~1.7 GB · 18,076 grayscale images · ~12,568 labelled · 4 classes · RLE-encoded masks)

## Phases

Each `phases/phaseN/README.md` is self-contained: goal, concrete steps, the exact
artifacts it must produce, and a **DONE test** that can be checked by someone else.
A phase is finished only when its DONE test passes — not when it "feels done".

| Phase | Name | Owner role | Blocks |
|---|---|---|---|
| 0 | Project setup + fixed data split | Classification+Lead | everyone |
| 1 | Data acquisition + RLE decode | Data+EDA | phases 3, 4 |
| 2 | EDA | Data+EDA | phase 3 (single- vs multi-label) |
| 3 | Classification (ResNet-50 vs EfficientNet-B2) | Classification+Lead | phase 6 |
| 4 | Segmentation (U-Net vs DeepLabV3+) | Segmentation+Ops | phase 6 |
| 5 | Model operations (deploy + maintenance) | Segmentation+Ops | phase 6 |
| 6 | Report + presentation | all | — |

## Layout

```
src/         rle.py, split.py, datasets.py, metrics.py
notebooks/   01_eda, 02a_classification, 02b_segmentation
splits/      train.csv / val.csv  ← generated once in phase 0, never edited after
data/        raw dataset + index.csv (gitignored)
outputs/     figs/ metrics/ ckpt/
report/      per-section markdown
phases/      phaseN/README.md — the executable plan for that phase
```

## Ground rules

- **One split for everyone.** `splits/*.csv` is generated once in phase 0 with a fixed
  seed. Every model in every phase reads it. Without this, the four models' metrics are
  not comparable and the whole champion-challenger comparison is void.
- **Metrics go to `outputs/metrics/*.json`**, not just printed in a notebook, so the
  report can be assembled from files rather than by re-running everything.
- **Grading criteria are quoted verbatim** in each phase README so nobody has to guess
  what earns the marks.
