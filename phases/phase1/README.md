# Phase 1 — Data acquisition + RLE decoding

**Owner** Data+EDA · **Day 1–2** · **Blocks: phase 3 and phase 4 entirely**

## Goal
Turn the Kaggle download into (a) a verified RLE decoder and (b) one index table that
every other phase reads.

## Steps
1. Accept the competition rules on the Kaggle website (required before download).
2. `kaggle competitions download -c severstal-steel-defect-detection` and unzip to `data/`.
3. **First thing, before any modelling: confirm the real image dimensions and aspect
   ratio.** These images are reportedly wide strips (possibly 1600×256) rather than
   squares — this has NOT been verified, and it determines the crop/resize strategy and
   how the pretrained backbone is attached. Print `Image.open(...).size` for 10 random
   files, write the answer into `phases/phase1/RESULT.md`, and tell the whole team.
4. Write `src/rle.py`:
   ```python
   def rle_decode(rle_str: str, shape: tuple[int, int]) -> np.ndarray  # -> 0/1 mask
   def rle_encode(mask: np.ndarray) -> str
   ```
   Mind the convention: Severstal RLE is **column-major (Fortran order)** with
   **1-based** start positions. Getting either wrong produces a plausible-looking but
   transposed or shifted mask.
5. Write the **round-trip test** `src/test_rle.py`: sample 20 rows from `train.csv`,
   `rle_encode(rle_decode(s)) == s` for all 20.
   This test is mandatory because **a wrong RLE decode raises no error** — it silently
   misplaces masks, and you would only discover it after phase 4 has trained.
6. Build `data/index.csv` with columns:
   `image_id, has_defect, class_id, rle, defect_area_px`
7. Sanity-view: overlay the decoded mask on 5 random defect images and confirm by eye the
   highlighted region sits on a visible surface flaw.

## Artifacts
- `src/rle.py`, `src/test_rle.py`, `data/index.csv`
- `phases/phase1/RESULT.md` recording the confirmed image size + row counts

## DONE test
- `python src/test_rle.py` → **20/20 pass**
- 5 overlay images saved to `outputs/figs/phase1_mask_overlay_*.png` and visually correct
- `data/index.csv` row count matches the number of `train.csv` annotation rows
