from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ModuleNotFoundError:
    pd = None
    HAS_PANDAS = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ModuleNotFoundError:
    torch = None
    nn = None
    HAS_TORCH = False


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cls_optimization import (  # noqa: E402
    calibrate_thresholds,
    forward_training_logits,
    horizontal_tile_starts,
    predict_probabilities,
    save_calibration,
)
try:
    from make_inner_split import create_inner_split  # noqa: E402
    HAS_TABULAR = True
except ModuleNotFoundError:
    create_inner_split = None
    HAS_TABULAR = False


if HAS_TORCH:
    class MeanModel(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            mean = images.mean(dim=(1, 2, 3), keepdim=False)
            return mean[:, None].repeat(1, 4)


class OptimizationTests(unittest.TestCase):
    def test_tiles_cover_full_width(self) -> None:
        starts = horizontal_tile_starts(1600, tile_width=896, overlap=192)
        self.assertEqual(starts, [0, 704])
        coverage = np.zeros(1600, dtype=np.uint8)
        for start in starts:
            coverage[start : start + 896] = 1
        self.assertTrue(coverage.all())

    @unittest.skipUnless(HAS_TORCH, "PyTorch model checks run in Colab")
    def test_tiled_training_and_tta_shapes(self) -> None:
        model = MeanModel()
        images = torch.randn(2, 3, 32, 160)
        logits = forward_training_logits(
            model, images, mode="tiles", tile_width=96, overlap=32
        )
        probs = predict_probabilities(
            model,
            images,
            mode="tiles",
            tile_width=96,
            overlap=32,
            horizontal_flip_tta=True,
        )
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(probs.shape), (2, 4))
        self.assertTrue(torch.all((probs >= 0) & (probs <= 1)))

    def test_per_class_calibration(self) -> None:
        labels = np.array(
            [
                [1, 0, 0, 1],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 0],
                [0, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
        probs = labels * 0.65 + (1 - labels) * 0.25
        result = calibrate_thresholds(labels, probs)
        self.assertEqual(len(result["thresholds"]), 4)
        self.assertTrue(all(0.25 < value <= 0.65 for value in result["thresholds"]))

    def test_calibration_refuses_outer_val(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "inner_calibration"):
                save_calibration(
                    {"thresholds": [0.5] * 4, "per_class": {}},
                    str(Path(tmp) / "thresholds.json"),
                    "outer_val",
                )

    @unittest.skipUnless(HAS_PANDAS and HAS_TABULAR, "pandas/sklearn checks run in Colab")
    def test_inner_split_partitions_only_outer_train(self) -> None:
        rows = []
        for index in range(120):
            labels = [0, 0, 0, 0]
            primary = 0
            if 60 <= index < 75:
                labels, primary = [1, 0, 0, 0], 1
            elif 75 <= index < 85:
                labels, primary = [0, 1, 0, 0], 2
            elif 85 <= index < 105:
                labels, primary = [0, 0, 1, 0], 3
            elif index >= 105:
                labels, primary = [0, 0, 0, 1], 4
            rows.append(
                {
                    "image_id": f"img_{index:03d}.jpg",
                    "has_defect": int(any(labels)),
                    "primary_class": primary,
                    **{f"has_class_{c + 1}": labels[c] for c in range(4)},
                }
            )
        frame = pd.DataFrame(rows)
        val_indices = set(range(10)) | {60, 75, 85, 86} | set(range(105, 111))
        outer_val = frame.iloc[sorted(val_indices)][["image_id"]]
        outer_train = frame.drop(index=sorted(val_indices))[["image_id"]]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            index_path = tmp_path / "index.csv"
            train_path = tmp_path / "train.csv"
            val_path = tmp_path / "val.csv"
            out_dir = tmp_path / "splits"
            frame.to_csv(index_path, index=False)
            outer_train.to_csv(train_path, index=False)
            outer_val.to_csv(val_path, index=False)

            manifest = create_inner_split(
                str(index_path),
                str(train_path),
                str(out_dir),
                str(val_path),
                calibration_fraction=0.2,
                seed=142,
            )
            inner_train = set(pd.read_csv(out_dir / "inner_train.csv")["image_id"])
            inner_cal = set(pd.read_csv(out_dir / "inner_calibration.csv")["image_id"])
            outer_train_ids = set(outer_train["image_id"])
            outer_val_ids = set(outer_val["image_id"])
            self.assertFalse(inner_train & inner_cal)
            self.assertEqual(inner_train | inner_cal, outer_train_ids)
            self.assertFalse((inner_train | inner_cal) & outer_val_ids)
            self.assertGreater(manifest["inner_calibration"]["per_class"]["2"], 0)
            stored = json.loads((out_dir / "inner_split_manifest.json").read_text())
            self.assertEqual(stored["seed"], 142)


if __name__ == "__main__":
    unittest.main()
