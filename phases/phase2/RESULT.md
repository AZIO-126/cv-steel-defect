# Phase 2 result — EDA (rubric item 2, 15 marks)

Figures: `outputs/figs/eda_structured.png`, `eda_intensity.png`, `eda_samples.png`,
`eda_outliers.png`. Schema table: `outputs/figs/schema_table.csv`. Code: `src/eda.py`.

## Rubric coverage
| Requirement (verbatim) | Where |
|---|---|
| "record counts, missing values, and schema" | printed counts + `schema_table.csv` (dtype / n_unique / example per column); `train.csv` missing-value tally |
| "Visualize the raw dataset using charts and tables" | `eda_structured.png` — per-class bar, defect/defect-free bar with %, defect-area violin per class (log10), class co-occurrence heatmap |
| "Image Data: Histograms, samples, outliers" | `eda_intensity.png` (global grayscale histogram + mean-intensity per class); `eda_samples.png` (one sample per class, original beside mask overlay); `eda_outliers.png` (darkest / brightest / largest-defect / smallest-defect) |

## Five report-ready conclusions
1. Binary has-defect is nearly balanced — 6,666 vs 5,902 (53.0% defective) — so **the binary
   task needs no resampling**.
2. The severe imbalance is **between defect classes**: class 3 has 5,150 images against class
   2's 247, a **21:1** ratio. Weighting/sampling belongs on the multi-class problem.
3. **427 of 6,666 defect images (6.41%) carry more than one class**, so classification must be
   **multi-label (4 sigmoids + BCE)**, not 4-way softmax. The co-occurrence heatmap shows the
   pairs concretely: class 3↔4 co-occur 284 times, class 1↔3 93 times.
4. Class 2 is **both rare and small-area** (violin plot), so it will be the hardest to segment.
   Report per-class Dice split by defect area rather than one pooled number.
5. Images are uniformly **1600×256 wide strips**, so a square-crop pipeline would throw away
   most of the frame — resize or tile along the width.

## Verification note
The mask overlays in `eda_samples.png` were inspected: the red regions sit on visible surface
flaws (class 1 spot cluster, class 2 vertical line). This is the geometric check that a
round-trip test alone cannot give — a transposed-but-self-consistent decode would pass the
round-trip and fail here.

## Provenance / what ran where
- Phases 0–1 (download, geometry, RLE round-trip, index.csv, frozen split) **executed in
  Colab** — see `phases/phase1/RESULT.md` for the captured output (1.68 GB in 18s,
  `W x H = (1600, 256)`, 20/20 round-trip, 12568/6666/5902, split 10054/2514).
- The same scripts were re-run locally and produced **identical numbers**, which is the
  reproducibility evidence.
- These EDA figures were rendered locally from that same verified `index.csv` + data.
  `src/eda.py` is environment-independent (matplotlib over index.csv + train_images), so a
  Colab Run-all reproduces them.
