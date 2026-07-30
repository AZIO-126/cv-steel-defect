# Phase 0 — Project setup + the one fixed data split

**Owner** Classification+Lead · **Day 1** · **Blocks: everyone**

## Goal
Everything downstream must be comparable. That requires one immutable train/val split
generated before anyone trains anything.

## Steps
1. `pip install -r requirements.txt` on Colab; confirm `torch.cuda.is_available()` on a T4.
2. Write `src/split.py`:
   - read `data/index.csv` (produced in phase 1 — for now accept a `--dry-run` that
     synthesises image ids so the script can be tested before data lands)
   - fixed `SEED = 42`
   - **stratify by `has_defect`** (and by `class_id` where present) so both splits carry
     the same defect prevalence
   - 80/20 train/val
   - write `splits/train.csv`, `splits/val.csv` with columns `image_id,split`
3. Commit the generated split files to git. **After this commit they are frozen.**
4. Write `src/datasets.py` with one `SteelDataset` class that both the classification and
   segmentation phases import (returns image, class label, and mask). One dataset class,
   not two — otherwise the two phases silently diverge on preprocessing.

## Artifacts
- `requirements.txt` (already present), `src/split.py`, `src/datasets.py`
- `splits/train.csv`, `splits/val.csv` — committed

## DONE test
- `python src/split.py --verify` prints the row counts of both splits and the defect
  prevalence in each, and the two prevalences differ by **< 1 percentage point**
- Both files are committed and their git hash is recorded in `phases/phase0/RESULT.md`
