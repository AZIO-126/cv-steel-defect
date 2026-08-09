# Data

The Severstal steel-defect dataset for this project.

## In this repo (git)
- `train.csv` — RLE-encoded defect masks (ImageId_ClassId, EncodedPixels).
- `index.csv` — the built index (per-image class flags + defect/clean), from `src/build_index.py`.
- `sample_submission.csv` — the Kaggle sample submission format.

## Full images (via GitHub Releases, not git)
The 1.6 GB of train/test images exceed GitHub's 100 MB per-file git limit, so they are attached as a
Release asset instead. Download `severstal.zip` from the repo's **Releases** page (tag `data-v2`),
then unzip into this `data/` folder so you have:

```
data/train_images/   (12,568 images, 1600x256 grayscale)
data/test_images/
data/train.csv
```

Source: Severstal Steel Defect Detection (Kaggle competition dataset).
