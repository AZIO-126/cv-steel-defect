# Phase 2 — Exploratory Data Analysis

**Owner** Data+EDA · **Day 3–4** · **Worth 15 marks** · **Blocks: phase 3's task definition**

## Grading criteria (verbatim from the assignment)
> Summarize data: record counts, missing values, and schema
> Visualize the raw dataset using charts and tables (line/bar charts, q-q plots, heatmaps, violin, etc.)
> Image Data: Histograms, samples, outliers

## Steps — each row below maps to one criterion
| Criterion | Produce |
|---|---|
| record counts / missing / schema | total images, labelled count, per-class counts, `train.csv` missing-value check, a DataFrame schema table |
| charts and tables | per-class defect count bar chart; stacked bar of defect vs no-defect (shows the class imbalance); violin plot of defect area; heatmap of multi-defect co-occurrence |
| image histograms / samples / outliers | grayscale intensity histogram (global + per class); sample grid per defect class (image + mask overlay); outliers — all-black / all-white / over-exposed frames, and extreme large/small defect areas |

## Two analyses to add beyond the checklist
1. **Defect area vs class** (bivariate) — this is what later justifies the "small defects
   are harder to segment" finding in phase 4.
2. **Multi-class co-occurrence rate per image** — **required, not optional.** If images
   frequently carry more than one defect class, phase 3 must be built as multi-label
   (4 sigmoid heads), not 4-way softmax. **Deliver this number by Day 3** so phase 3 does
   not build the wrong head.

## Artifacts
- `notebooks/01_eda.ipynb`, all figures in `outputs/figs/`
- `phases/phase2/RESULT.md` — 5 EDA conclusions written as report-ready sentences, plus
  the single-vs-multi-label recommendation stated explicitly

## DONE test
- Every row of the criteria table above has a corresponding saved figure or table
- The single-vs-multi-label question has a one-line answer with the co-occurrence number
  behind it
