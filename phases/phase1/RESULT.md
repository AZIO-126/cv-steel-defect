# Phase 1 result

## Confirmed image geometry (was previously an unverified assumption)
- A real file (`test_images/0000f269f.jpg`, 106,993 bytes) reports **1600 x 256 (W x H)**.
- Cross-check: the full-image mask in `sample_submission.csv` is `1 409600`, and
  256 * 1600 = 409,600. Independently, the maximum pixel position across every RLE run in
  `train.csv` is exactly 409,600.
- Note 409,600 alone is ambiguous (it also factors as 128x3200 and 640x640), which is why
  the actual file was fetched rather than inferred. **`rle_decode` must be called with
  `shape=(256, 1600)`** — the codec is column-major, so a transposed shape silently yields
  transposed masks.

## Kaggle API notes
- Auth: `KAGGLE_API_TOKEN` (the `KGAT_...` form) works as an HTTP `Authorization: Bearer`
  header. The anaconda py3.9 `kaggle` CLI on this machine is too old to accept it.
- Nested file downloads need the slash URL-encoded:
  `.../data/download/severstal-steel-defect-detection/test_images%2F0000f269f.jpg`
  The un-encoded form returns 404 with an HTML body.
- Downloading anything requires having joined the competition, which in turn requires
  **phone verification** (Persona). Listing file metadata does NOT require it — so a 200 on
  `data/list` is not evidence that downloads will work.

## train.csv shape
- 7,095 annotation rows, columns `ImageId, ClassId, EncodedPixels`
  (one row per defect instance, so multi-defect images appear more than once and
  defect-free images are absent entirely).

## RLE round-trip test
`python src/test_rle.py --csv data/train.csv` → **9/9 pass**, including 20 real rows.

## index.csv (defect images only, pending the image download)
- 6,666 images carry at least one defect.
- Per-class image counts: class 1 = 897, class 2 = 247, class 3 = 5,150, class 4 = 801.
  **Class 3 outnumbers class 2 by roughly 21:1.**
- Defect classes per image: 6,239 have one, 425 have two, 2 have three.

## Answer to the question blocking phase 3
**427 of 6,666 defect images (6.41%) carry more than one defect class**, so the
classification head must be **multi-label (4 sigmoid outputs + BCE)**, not 4-way softmax.
A softmax head would force a single label onto 6.41% of the training data and make those
images unlearnable by construction.

## Still to do (needs the 1.7 GB image download)
- Re-run `build_index.py --images-dir` so the defect-free images enter the index; only then
  is the real class balance known and the split final.
- Regenerate and freeze `splits/*.csv` from that complete index.
