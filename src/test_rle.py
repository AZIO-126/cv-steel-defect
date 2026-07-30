"""Round-trip test for the Severstal RLE codec.

Mandatory before any segmentation work: a wrong RLE decode raises no exception, it just
silently misplaces the mask, and the mistake would only surface after a model had trained.

Runs without the dataset (synthetic masks + the column-major convention checks). Once
train.csv is available, `--csv path/to/train.csv` additionally round-trips 20 real rows,
which is the phase-1 DONE test.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rle import defect_area_px, rle_decode, rle_encode  # noqa: E402

H, W = 256, 1600  # Severstal's reported shape; confirm against real files in phase 1


def check(name: str, ok: bool) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="path to train.csv, to round-trip 20 real rows")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    results = []

    # random masks round-trip
    bad = 0
    for _ in range(200):
        m = np.zeros((H, W), np.uint8)
        for _ in range(rng.integers(1, 6)):
            r0, c0 = rng.integers(0, H - 30), rng.integers(0, W - 60)
            m[r0:r0 + rng.integers(3, 30), c0:c0 + rng.integers(3, 60)] = 1
        if not np.array_equal(rle_decode(rle_encode(m), (H, W)), m):
            bad += 1
    results.append(check(f"200 random masks round-trip ({200 - bad}/200)", bad == 0))

    # string round-trip, including runs touching the first and last pixel
    strs = ["1 4", "5 2 12 3", f"{H * W - 3} 3", "1 1", f"1 {H * W}"]
    results.append(check("string -> mask -> string is byte-identical",
                         all(rle_encode(rle_decode(s, (H, W))) == s for s in strs)))

    # the column-major trap: vertically adjacent pixels are ONE run, horizontally adjacent are TWO
    v = np.zeros((H, W), np.uint8); v[0, 0] = v[1, 0] = 1
    h = np.zeros((H, W), np.uint8); h[0, 0] = h[0, 1] = 1
    results.append(check("column-major: vertical neighbours form one run",
                         rle_encode(v) == "1 2"))
    results.append(check("column-major: horizontal neighbours are separated by H",
                         rle_encode(h) == f"1 1 {H + 1} 1"))

    # absent-class representations
    results.append(check("None / NaN / '' decode to an empty mask",
                         all(rle_decode(x, (H, W)).sum() == 0
                             for x in (None, float("nan"), ""))))
    results.append(check("empty mask encodes to the empty string",
                         rle_encode(np.zeros((H, W), np.uint8)) == ""))

    # area helper agrees with the materialised mask
    s = "5 10 100 20"
    results.append(check("defect_area_px agrees with mask.sum()",
                         defect_area_px(s) == rle_decode(s, (H, W)).sum()))

    # overflow must raise rather than truncate
    try:
        rle_decode(f"{H * W} 5", (H, W))
        results.append(check("out-of-range run raises ValueError", False))
    except ValueError:
        results.append(check("out-of-range run raises ValueError", True))

    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv).dropna(subset=["EncodedPixels"])
        sample = df.sample(min(20, len(df)), random_state=0)
        ok = sum(rle_encode(rle_decode(r.EncodedPixels, (H, W))) == r.EncodedPixels.strip()
                 for r in sample.itertuples())
        results.append(check(f"20 real train.csv rows round-trip ({ok}/{len(sample)})",
                             ok == len(sample)))
    else:
        print("  SKIP  real train.csv rows (pass --csv once the data is downloaded)")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
