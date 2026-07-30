"""Generate the ONE fixed train/val split every model in every phase must use.

Why this exists as a frozen artifact rather than a call inside each notebook: the rubric's
champion-challenger comparison is only meaningful if all four models are scored on identical
data. If each notebook splits independently, ResNet-50 and U-Net are evaluated on different
validation sets and none of the numbers can be compared.

Stratification is on the joint key (has_defect, primary_class) so that both the defect
prevalence AND the very skewed class mix are preserved in both halves. Class 2 has only ~247
images against class 3's ~5150, so a naive random split can easily leave the validation set
with too few class-2 examples to score.

Run AFTER data/index.csv includes the defect-free images (build_index.py --images-dir),
then commit splits/*.csv and do not regenerate them.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
VAL_FRACTION = 0.2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--outdir", default="splits")
    ap.add_argument("--verify", action="store_true",
                    help="check an existing split instead of regenerating it")
    args = ap.parse_args()

    train_p = os.path.join(args.outdir, "train.csv")
    val_p = os.path.join(args.outdir, "val.csv")

    if args.verify:
        if not (os.path.exists(train_p) and os.path.exists(val_p)):
            print("  no split to verify — run without --verify first")
            return 1
        idx = pd.read_csv(args.index).set_index("image_id")
        tr = pd.read_csv(train_p)
        va = pd.read_csv(val_p)
        print(f"  train rows: {len(tr)}   val rows: {len(va)}")
        overlap = set(tr.image_id) & set(va.image_id)
        print(f"  overlap between train and val: {len(overlap)}  {'OK' if not overlap else 'FAIL'}")
        rates = {}
        for name, part in (("train", tr), ("val", va)):
            sub = idx.loc[idx.index.intersection(part.image_id)]
            rates[name] = 100.0 * sub.has_defect.mean() if len(sub) else float("nan")
            print(f"  {name:5s} defect prevalence: {rates[name]:.2f}%")
        gap = abs(rates["train"] - rates["val"])
        print(f"  prevalence gap: {gap:.3f} pp  "
              f"{'OK (< 1 pp)' if gap < 1.0 else 'FAIL (>= 1 pp)'}")
        print("\n  per-class image counts:")
        for c in range(1, 5):
            t = int(idx.loc[idx.index.intersection(tr.image_id), f"has_class_{c}"].sum())
            v = int(idx.loc[idx.index.intersection(va.image_id), f"has_class_{c}"].sum())
            print(f"    class {c}: train {t:5d}   val {v:5d}")
        return 0 if (not overlap and gap < 1.0) else 1

    idx = pd.read_csv(args.index)
    # Joint stratification key: presence of a defect AND which class dominates.
    strat = idx["has_defect"].astype(str) + "_" + idx["primary_class"].astype(str)
    # Any stratum with a single member cannot be split; fold those into a catch-all.
    counts = strat.value_counts()
    strat = strat.where(strat.map(counts) >= 2, "rare")

    tr, va = train_test_split(idx["image_id"], test_size=VAL_FRACTION,
                              random_state=SEED, stratify=strat, shuffle=True)
    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame({"image_id": sorted(tr), "split": "train"}).to_csv(train_p, index=False)
    pd.DataFrame({"image_id": sorted(va), "split": "val"}).to_csv(val_p, index=False)
    print(f"  wrote {train_p} ({len(tr)}) and {val_p} ({len(va)}) with SEED={SEED}")
    print("  now run:  python src/split.py --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
