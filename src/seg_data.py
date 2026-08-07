"""Dataset + loaders for phase 4 segmentation.

One image -> a 4-channel 0/1 mask, channel c-1 holding class c. Multi-label rather than a
single 5-way label map, because phase 1 measured that 6.41% of defect images carry more than
one class; a single argmax label map would have to throw one of them away.

Images are 1600x256 and stay that way. Two reasons not to resize:
  - 1600 and 256 are both divisible by 32, so every smp encoder accepts the native frame and
    no resizing is needed to make the architecture happy in the first place;
  - phase 2 found class 2 defects are both rare and small-area, and downscaling a thin
    scratch is exactly how those disappear before the model ever sees them.

Masks come from `src/rle.py`, which phase 1 verified round-trips on real rows and is
column-major. Nothing here re-implements RLE.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rle import rle_decode  # noqa: E402

IMAGE_SHAPE = (256, 1600)  # (H, W) — verified in phase 1, see src/build_index.py
N_CLASSES = 4

# ImageNet statistics. The images are grayscale and get replicated to 3 channels (see
# below), so the same scalar mean/std applies to all three.
IMAGENET_MEAN = 0.449
IMAGENET_STD = 0.226


def load_masks_table(train_csv: str) -> dict[str, dict[int, str]]:
    """Kaggle train.csv -> {image_id: {class_id: rle_string}}.

    train.csv holds one row per defect instance, so an image with two classes appears twice
    and a defect-free image does not appear at all. Absent entries mean an empty mask, which
    is what `rle_decode(None)` already returns.
    """
    df = pd.read_csv(train_csv)
    table: dict[str, dict[int, str]] = {}
    for row in df.itertuples():
        rle = row.EncodedPixels
        if rle is None or (isinstance(rle, float) and np.isnan(rle)):
            continue
        table.setdefault(str(row.ImageId), {})[int(row.ClassId)] = str(rle)
    return table


def build_mask(rles: dict[int, str] | None, shape: tuple[int, int] = IMAGE_SHAPE) -> np.ndarray:
    """4-channel (C, H, W) 0/1 mask. Channel index c-1 carries class c."""
    height, width = shape
    mask = np.zeros((N_CLASSES, height, width), dtype=np.uint8)
    if not rles:
        return mask
    for class_id, rle in rles.items():
        mask[class_id - 1] = rle_decode(rle, shape)
    return mask


def augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Flips plus brightness/contrast jitter — deliberately mild.

    Horizontal and vertical flips are safe here: a steel strip has no up or down, and the
    defect classes are distinguished by texture and shape, not orientation. Rotation and
    elastic warping are left out on purpose (phase 3's README makes the same call): they
    smear thin class-2 scratches into something the label no longer describes.

    Written with numpy rather than albumentations so the pipeline has no OpenCV dependency —
    these four operations are the whole augmentation budget and are exact in numpy.
    """
    if rng.random() < 0.5:                      # horizontal flip
        image = image[:, ::-1]
        mask = mask[:, :, ::-1]
    if rng.random() < 0.5:                      # vertical flip
        image = image[::-1, :]
        mask = mask[:, ::-1, :]
    if rng.random() < 0.5:                      # brightness / contrast jitter
        alpha = 1.0 + rng.uniform(-0.2, 0.2)    # contrast
        beta = rng.uniform(-0.1, 0.1)           # brightness, in [0,1] image units
        image = np.clip(image * alpha + beta, 0.0, 1.0)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class SteelSegDataset(Dataset):
    """Yields (image[3,H,W] float32, mask[4,H,W] float32) for one image id.

    The image is grayscale; it is replicated across 3 channels so the ImageNet-pretrained
    encoder can be used unmodified. The alternative — collapsing the first conv to 1 channel
    — throws away the pretrained filters' colour structure for no gain on a dataset this
    small, and phase 3 makes the same choice, which keeps the two problems comparable.
    """

    def __init__(
        self,
        image_ids: list[str],
        images_dir: str,
        masks_table: dict[str, dict[int, str]],
        train: bool = False,
        seed: int = 42,
    ):
        self.image_ids = list(image_ids)
        self.images_dir = images_dir
        self.masks_table = masks_table
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, i: int):
        image_id = self.image_ids[i]
        with Image.open(os.path.join(self.images_dir, image_id)) as img:
            image = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        if image.shape != IMAGE_SHAPE:
            raise ValueError(f"{image_id}: expected {IMAGE_SHAPE}, got {image.shape}")

        mask = build_mask(self.masks_table.get(image_id), IMAGE_SHAPE)

        if self.train:
            # Seeded per (epoch-independent) index so a rerun of the same worker is
            # reproducible; torch's own seeding covers the epoch-to-epoch variation.
            rng = np.random.default_rng(self.seed + i + int(torch.randint(0, 1 << 30, (1,))))
            image, mask = augment(image, mask, rng)

        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.repeat(image[None, :, :], 3, axis=0)  # 1 channel -> 3
        return torch.from_numpy(image.astype(np.float32)), torch.from_numpy(mask.astype(np.float32))


def read_split(splits_dir: str, name: str) -> list[str]:
    """Read the frozen phase-0 split. Never regenerate it — every phase reads these files."""
    path = os.path.join(splits_dir, f"{name}.csv")
    return pd.read_csv(path)["image_id"].astype(str).tolist()


def make_loaders(
    data_dir: str,
    splits_dir: str,
    batch_size: int = 8,
    num_workers: int = 2,
    seed: int = 42,
    limit: int | None = None,
) -> tuple[DataLoader, DataLoader, dict[str, dict[int, str]]]:
    images_dir = os.path.join(data_dir, "train_images")
    masks_table = load_masks_table(os.path.join(data_dir, "train.csv"))

    train_ids = read_split(splits_dir, "train")
    val_ids = read_split(splits_dir, "val")
    if limit:
        train_ids, val_ids = train_ids[:limit], val_ids[:limit]

    train_ds = SteelSegDataset(train_ids, images_dir, masks_table, train=True, seed=seed)
    val_ds = SteelSegDataset(val_ids, images_dir, masks_table, train=False, seed=seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, masks_table
