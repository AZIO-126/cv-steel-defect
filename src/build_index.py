"""Build data/index.csv — the single table every later phase reads.

Severstal's `train.csv` holds ONE ROW PER DEFECT INSTANCE (ImageId, ClassId, EncodedPixels),
so an image with two defect classes appears twice and a defect-free image does not appear at
all. Phases 3 and 4 need a per-image view instead, including the defect-free images, so this
script pivots train.csv into one row per image and adds the derived columns the EDA and the
models rely on.

IMAGE_SHAPE is (256, 1600) — verified against a real file, not assumed: a downloaded
train/test jpg reports 1600x256 (W x H), and the full-image mask in sample_submission.csv is
`1 409600` = 256 * 1600. This matters because rle_decode is column-major, so passing the
transposed shape would silently produce transposed masks.

Usage:
    python src/build_index.py                       # needs data/train.csv
    python src/build_index.py --images-dir data/train_images   # also records per-image size
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rle import defect_area_px  # noqa: E402

IMAGE_SHAPE = (256, 1600)  # (H, W) — verified, see module docstring
N_CLASSES = 4


def build(train_csv: str, images_dir: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    expected = {"ImageId", "ClassId", "EncodedPixels"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {missing}")

    df["defect_area_px"] = df["EncodedPixels"].map(defect_area_px)

    # One row per (image, class) -> one row per image.
    rows = []
    for image_id, g in df.groupby("ImageId", sort=True):
        classes = sorted(int(c) for c in g["ClassId"].unique())
        per_class_area = {int(r.ClassId): int(r.defect_area_px) for r in g.itertuples()}
        rows.append({
            "image_id": image_id,
            "has_defect": 1,
            "n_defect_classes": len(classes),
            "class_ids": "|".join(str(c) for c in classes),
            # primary_class = the class with the largest area, used for single-label baselines
            "primary_class": max(per_class_area, key=per_class_area.get),
            "defect_area_px": int(g["defect_area_px"].sum()),
            **{f"area_class_{c}": per_class_area.get(c, 0) for c in range(1, N_CLASSES + 1)},
            **{f"has_class_{c}": int(c in per_class_area) for c in range(1, N_CLASSES + 1)},
        })
    idx = pd.DataFrame(rows)

    # Defect-free images exist only on disk, never in train.csv. Without them the class
    # imbalance disappears and every downstream metric is measured on the wrong population.
    if images_dir and os.path.isdir(images_dir):
        on_disk = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(".jpg"))
        clean = sorted(set(on_disk) - set(idx["image_id"]))
        if clean:
            blank = pd.DataFrame({
                "image_id": clean, "has_defect": 0, "n_defect_classes": 0,
                "class_ids": "", "primary_class": 0, "defect_area_px": 0,
                **{f"area_class_{c}": 0 for c in range(1, N_CLASSES + 1)},
                **{f"has_class_{c}": 0 for c in range(1, N_CLASSES + 1)},
            })
            idx = pd.concat([idx, blank], ignore_index=True)
        print(f"  images on disk: {len(on_disk)} | defect-free: {len(clean)}")
    else:
        print("  NOTE: no images dir given — index covers only the DEFECT images from "
              "train.csv. Re-run with --images-dir once the images are downloaded so the "
              "defect-free images (and therefore the real class balance) are included.")

    return idx.sort_values("image_id").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/train.csv")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--out", default="data/index.csv")
    args = ap.parse_args()

    idx = build(args.train_csv, args.images_dir)
    idx.to_csv(args.out, index=False)

    print(f"\n  wrote {args.out}: {len(idx)} rows")
    print(f"  images with a defect : {int((idx.has_defect == 1).sum())}")
    print(f"  images defect-free   : {int((idx.has_defect == 0).sum())}")
    print("\n  per-class image counts:")
    for c in range(1, N_CLASSES + 1):
        n = int(idx[f"has_class_{c}"].sum())
        print(f"    class {c}: {n:5d} images")
    print("\n  defect classes per image:")
    print(idx.n_defect_classes.value_counts().sort_index().to_string())
    multi = int((idx.n_defect_classes > 1).sum())
    defect = int((idx.has_defect == 1).sum())
    print(f"\n  multi-class images: {multi} / {defect} defect images "
          f"= {100.0 * multi / max(defect, 1):.2f}%")
    print("  -> phase 3 head choice: "
          + ("MULTI-LABEL (4 sigmoids)" if multi / max(defect, 1) > 0.02
             else "single-label softmax is defensible"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
