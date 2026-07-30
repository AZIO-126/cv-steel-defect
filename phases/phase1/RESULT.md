# Phase 1 result — verified in Colab, cross-checked locally

Colab is the delivery environment, so the numbers below were produced there first and then
reproduced byte-for-byte by the repo scripts on a laptop. Identical output from two
independent environments is the reproducibility evidence.

## Image geometry — measured, not assumed
`W x H = (1600, 256) | total px 409600 | rle shape (H,W) = (256, 1600)`

Read off 20 random real files (all identical). Necessary because 409,600 also factors as
128x3200 and 640x640, and `rle_decode` is **column-major** — a transposed shape yields
transposed masks and raises nothing.

## RLE codec
`RLE round-trip on 20 real rows: 20/20 PASS`
`column-major check: True and True` — vertically adjacent pixels form ONE run (`1 2`);
horizontally adjacent are separated by H (`1 1 257 1`). Getting this backwards is silent.

## index.csv — the true class balance
```
12568 rows | defect 6666 | clean 5902 (47.0% clean)
  class 1:   897     class 2:   247
  class 3:  5150     class 4:   801
multi-class images: 427/6666 = 6.41%
```
`train.csv` has 7,095 rows (one per defect instance) and contains **no** defect-free images —
those exist only on disk, so they are folded in from `train_images/`.

## Two findings that change how phases 3 and 4 should be built
1. **The classification head must be MULTI-LABEL** (4 sigmoids + BCE), not 4-way softmax.
   6.41% of defect images carry more than one class; softmax would force a single label onto
   427 images and make them unlearnable by construction.
2. **The imbalance is NOT defect-vs-clean.** Binary has-defect is 53/47 — essentially
   balanced, so it needs no special handling. The severe skew is *between defect classes*:
   class 3 (5,150) vs class 2 (247), about **21:1**. Weighting / sampling belongs on the
   4-class problem, not the binary one. (The earlier plan's "lots of defect-free images"
   framing was wrong — corrected here from measurement.)

## Frozen split
```
train 10054 / val 2514 | SEED=42
defect prevalence  train 53.03%  val 53.06%  gap 0.029pp  OK (<1pp)
  class 1: train  718  val  179      class 2: train  200  val   47
  class 3: train 4116  val 1034      class 4: train  644  val  157
```
Stratified on the joint key (has_defect, primary_class) so both the prevalence and the skewed
class mix survive. Class 2 retains 47 validation images — enough to score.

## Kaggle / Colab notes worth not re-learning
- Auth: `KAGGLE_API_TOKEN` (`KGAT_...`) as an HTTP `Authorization: Bearer` header.
- **A 200 from `data/list` does NOT mean downloads work** — listing metadata needs no rules
  acceptance, downloading does. Probe a real file.
- Joining the competition requires **phone verification** (Persona), not just a rules click.
- Nested single-file downloads need the slash URL-encoded: `train_images%2Fxxx.jpg`.
- **Do not use `drive.mount()`**: it fails on this environment with
  `ValueError: mount failed` (Colab FAQ drive-timeout), and mounting would make the notebook
  non-self-contained. Download into `/content` instead — 1.68 GB in 18s, and any session can
  re-fetch it. Only the small artifacts (index.csv, splits, figures) belong in git.
- Colab Secrets: the per-row **Notebook access** toggle defaults to off; `userdata.get()` then
  raises `SecretNotFoundError`, and if the grant lands late you get
  `TimeoutException: Secrets can only be fetched when running from the Colab UI` — re-run
  after granting.
