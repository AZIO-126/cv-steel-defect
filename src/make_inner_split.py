"""Create an inner calibration split without touching the frozen report split.

The outer validation file is only used as a leakage guard. Every row written by
this script comes from ``splits/train.csv``. Thresholds and experiment choices
must be made on ``inner_calibration.csv`` and evaluated once on outer val later.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_COLUMNS = [f"has_class_{c}" for c in range(1, 5)]


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image_ids(path: str | os.PathLike[str]) -> pd.Series:
    frame = pd.read_csv(path)
    if "image_id" not in frame.columns:
        raise ValueError(f"{path} must contain an image_id column")
    if frame["image_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate image_id rows")
    return frame["image_id"].astype(str)


def _stratification_key(frame: pd.DataFrame, calibration_fraction: float) -> pd.Series:
    """Preserve multi-label patterns when possible and fold tiny strata safely."""
    pattern = frame[LABEL_COLUMNS].astype(int).astype(str).agg("".join, axis=1)
    key = "labels_" + pattern

    # A stratum needs enough rows to put at least one example on both sides.
    min_count = max(2, math.ceil(1.0 / calibration_fraction))
    counts = key.value_counts()
    fallback = (
        "primary_"
        + frame["has_defect"].astype(int).astype(str)
        + "_"
        + frame["primary_class"].fillna(0).astype(int).astype(str)
    )
    key = key.where(key.map(counts) >= min_count, fallback)

    # Extremely rare primary-class strata fall back to defect presence.
    counts = key.value_counts()
    key = key.where(
        key.map(counts) >= 2,
        "defect_" + frame["has_defect"].astype(int).astype(str),
    )
    return key


def create_inner_split(
    index_path: str,
    outer_train_path: str,
    out_dir: str,
    outer_val_path: str | None = None,
    calibration_fraction: float = 0.2,
    seed: int = 142,
) -> dict:
    if not 0.05 <= calibration_fraction <= 0.5:
        raise ValueError("calibration_fraction must be between 0.05 and 0.5")

    index = pd.read_csv(index_path)
    required = {"image_id", "has_defect", "primary_class", *LABEL_COLUMNS}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"index is missing columns: {sorted(missing)}")
    index["image_id"] = index["image_id"].astype(str)
    if index["image_id"].duplicated().any():
        raise ValueError("index contains duplicate image_id rows")

    outer_train_ids = _read_image_ids(outer_train_path)
    missing_from_index = sorted(set(outer_train_ids) - set(index["image_id"]))
    if missing_from_index:
        raise ValueError(
            f"{len(missing_from_index)} outer-train IDs are absent from index; "
            f"first: {missing_from_index[:3]}"
        )

    if outer_val_path:
        outer_val_ids = set(_read_image_ids(outer_val_path))
        overlap = set(outer_train_ids) & outer_val_ids
        if overlap:
            raise ValueError(
                "outer train and outer val overlap; refusing to create an inner split"
            )
    else:
        outer_val_ids = set()

    keyed = index.set_index("image_id", drop=False).loc[outer_train_ids].copy()
    stratify = _stratification_key(keyed, calibration_fraction)
    inner_train, inner_calibration = train_test_split(
        keyed["image_id"],
        test_size=calibration_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    inner_train = sorted(inner_train.astype(str))
    inner_calibration = sorted(inner_calibration.astype(str))
    if set(inner_train) & set(inner_calibration):
        raise AssertionError("inner train/calibration overlap")
    if set(inner_train) | set(inner_calibration) != set(outer_train_ids):
        raise AssertionError("inner split does not exactly partition outer train")
    if (set(inner_train) | set(inner_calibration)) & outer_val_ids:
        raise AssertionError("outer validation leakage detected")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_out = out / "inner_train.csv"
    calibration_out = out / "inner_calibration.csv"
    pd.DataFrame({"image_id": inner_train, "split": "inner_train"}).to_csv(
        train_out, index=False
    )
    pd.DataFrame(
        {"image_id": inner_calibration, "split": "inner_calibration"}
    ).to_csv(calibration_out, index=False)

    def counts(ids: list[str]) -> dict:
        part = index[index["image_id"].isin(ids)]
        return {
            "rows": int(len(part)),
            "defect_images": int(part["has_defect"].sum()),
            "per_class": {
                str(c): int(part[f"has_class_{c}"].sum()) for c in range(1, 5)
            },
        }

    manifest = {
        "purpose": "model selection and per-class threshold calibration only",
        "leakage_rule": "outer val is excluded and must not be used for tuning",
        "seed": seed,
        "calibration_fraction": calibration_fraction,
        "sources": {
            "index": {"path": index_path, "sha256": _sha256(index_path)},
            "outer_train": {
                "path": outer_train_path,
                "sha256": _sha256(outer_train_path),
            },
            "outer_val": (
                {"path": outer_val_path, "sha256": _sha256(outer_val_path)}
                if outer_val_path
                else None
            ),
        },
        "inner_train": counts(inner_train),
        "inner_calibration": counts(inner_calibration),
    }
    manifest_path = out / "inner_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index.csv")
    parser.add_argument("--outer-train", default="splits/train.csv")
    parser.add_argument("--outer-val", default="splits/val.csv")
    parser.add_argument("--out-dir", default="splits")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=142)
    args = parser.parse_args()

    manifest = create_inner_split(
        index_path=args.index,
        outer_train_path=args.outer_train,
        outer_val_path=args.outer_val,
        out_dir=args.out_dir,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    print("\nWrote splits/inner_train.csv and splits/inner_calibration.csv")
    print("Tune only on inner_calibration; evaluate outer val once after freezing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
