"""Run-length encoding for Severstal steel-defect masks.

Severstal's `train.csv` stores each defect mask as a run-length string of
space-separated `start length` pairs. Two conventions matter and both are easy to get
silently wrong — a mistake here raises no error, it just puts the mask in the wrong place,
and you would not find out until a segmentation model had already trained on it:

1. **Column-major (Fortran) order.** Pixels are numbered top-to-bottom down column 0,
   then down column 1, and so on — not left-to-right along each row.
2. **1-based start positions.** The first pixel of the image is position 1, not 0.

`rle_decode` and `rle_encode` are exact inverses on any mask, which is what
`test_rle.py` checks by round-tripping real rows out of `train.csv`.
"""
from __future__ import annotations

import numpy as np


def rle_decode(rle: str | float | None, shape: tuple[int, int]) -> np.ndarray:
    """Decode a Severstal RLE string into a 0/1 mask of `shape` = (height, width).

    An empty / NaN / None value decodes to an all-zero mask, which is what a defect-free
    image (or a defect class absent from a given image) should produce.
    """
    height, width = shape
    mask = np.zeros(height * width, dtype=np.uint8)

    # train.csv leaves the cell empty when a class is absent; pandas turns that into NaN.
    if rle is None or (isinstance(rle, float) and np.isnan(rle)):
        return mask.reshape(shape, order="F")
    rle = str(rle).strip()
    if not rle:
        return mask.reshape(shape, order="F")

    values = rle.split()
    if len(values) % 2 != 0:
        raise ValueError(f"RLE must hold an even number of tokens, got {len(values)}")

    starts = np.asarray(values[0::2], dtype=np.int64) - 1  # 1-based -> 0-based
    lengths = np.asarray(values[1::2], dtype=np.int64)

    if (starts < 0).any():
        raise ValueError("RLE start positions must be >= 1 (they are 1-based)")
    if (lengths <= 0).any():
        raise ValueError("RLE run lengths must be positive")
    ends = starts + lengths
    if (ends > mask.size).any():
        raise ValueError(
            f"RLE run overflows the image: max end {ends.max()} > {mask.size} pixels. "
            f"Wrong shape? Called with height={height}, width={width}."
        )

    for start, end in zip(starts, ends):
        mask[start:end] = 1

    # Fortran order: the flat index runs down columns, not across rows.
    return mask.reshape(shape, order="F")


def rle_encode(mask: np.ndarray) -> str:
    """Encode a 0/1 (or boolean) mask back into a Severstal RLE string.

    Exact inverse of `rle_decode`. An all-zero mask encodes to the empty string, matching
    how train.csv represents an absent defect class.
    """
    pixels = np.asarray(mask).flatten(order="F")
    if pixels.dtype != np.uint8:
        pixels = (pixels > 0).astype(np.uint8)

    # Sentinel zeros at both ends turn every run boundary into a 0->1 or 1->0 transition,
    # so np.diff finds them without special-casing masks that touch the first/last pixel.
    padded = np.concatenate([[0], pixels, [0]])
    transitions = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    starts = transitions[0::2]
    ends = transitions[1::2]
    lengths = ends - starts

    return " ".join(f"{s} {l}" for s, l in zip(starts, lengths))


def defect_area_px(rle: str | float | None) -> int:
    """Total defect pixel count straight from the RLE, without building the mask.

    Used to build `index.csv` over ~12.5k rows, where materialising every mask would be
    wasteful. Reads the length of each run and sums them.
    """
    if rle is None or (isinstance(rle, float) and np.isnan(rle)):
        return 0
    rle = str(rle).strip()
    if not rle:
        return 0
    return int(sum(int(v) for v in rle.split()[1::2]))
