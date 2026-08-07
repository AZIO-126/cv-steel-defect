"""Four-panel qualitative comparison figures: original / ground truth / U-Net / DeepLabV3+.

Both models are loaded from their saved checkpoints and run on the SAME images, so the two
prediction panels differ only by architecture. The images are picked deterministically to
cover every defect class plus the two area groups and the defect-free case, rather than at
random — a random draw over this dataset returns class 3 almost every time (5,150 images
against class 2's 247) and would show nothing about where the two models actually differ.

    python src/seg_figs.py --data-dir /content/steel --n-per-group 2
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seg_models  # noqa: E402
from seg_data import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SHAPE, build_mask, load_masks_table, read_split  # noqa: E402
from seg_metrics import postprocess  # noqa: E402

# One colour per defect class, kept identical across the GT and both prediction panels so a
# colour change between panels always means a class disagreement and never a palette change.
CLASS_COLOURS = {
    1: (0.90, 0.10, 0.10),   # red
    2: (0.10, 0.75, 0.20),   # green
    3: (0.15, 0.45, 0.95),   # blue
    4: (0.95, 0.75, 0.05),   # amber
}


def overlay(gray: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Tint the grayscale strip where each class channel is set."""
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    for c in range(mask.shape[0]):
        where = mask[c].astype(bool)
        if not where.any():
            continue
        colour = np.array(CLASS_COLOURS[c + 1], dtype=np.float32)
        rgb[where] = (1 - alpha) * rgb[where] + alpha * colour
    return np.clip(rgb, 0, 1)


def load_model(name: str, ckpt_dir: str, device):
    path = os.path.join(ckpt_dir, f"{name}_{seg_models.ENCODER}.pth")
    model = seg_models.build(name, pretrained=False).to(device)
    model.load_state_dict(torch.load(path, map_location=device)["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, gray: np.ndarray, device, threshold=0.5, min_area=300) -> np.ndarray:
    x = (gray - IMAGENET_MEAN) / IMAGENET_STD
    x = np.repeat(x[None, None, :, :], 3, axis=1).astype(np.float32)
    probs = torch.sigmoid(model(torch.from_numpy(x).to(device))).cpu().numpy()[0]
    return postprocess(probs, threshold, min_area)


def choose_images(index_csv: str, val_ids: list[str], n_per_group: int = 2) -> list[tuple[str, str]]:
    """Deterministic, coverage-driven selection -> list of (image_id, why it was chosen).

    Groups: each of the four classes (largest instance, so the figure is readable), the
    smallest-area defects (where U-Net's skip connections should show), a multi-class image,
    and a defect-free image (where the false-positive behaviour is visible).
    """
    index = pd.read_csv(index_csv)
    index = index[index.image_id.isin(set(val_ids))].sort_values("image_id").reset_index(drop=True)
    chosen: list[tuple[str, str]] = []
    taken: set[str] = set()

    def take(rows, reason, n=n_per_group) -> int:
        got = 0
        for image_id in rows.image_id.tolist():
            if image_id in taken:
                continue
            chosen.append((image_id, reason))
            taken.add(image_id)
            got += 1
            if got == n:
                break
        return got

    # Scarce groups pick first. Multi-class and smallest-defect images are a thin slice of
    # the validation set and are often ALSO the largest instance of some class, so if the
    # per-class groups ran first they would consume the only candidates and these groups
    # would come back empty — which is how a figure set silently loses its most interesting
    # cases.
    groups = [
        (index[index.n_defect_classes > 1].sort_values("defect_area_px", ascending=False),
         "multiple defect classes in one image"),
        (index[(index.has_defect == 1) & (index.defect_area_px > 0)].sort_values("defect_area_px"),
         "smallest defect area in val"),
    ]
    for c in range(1, 5):
        groups.append((
            index[index[f"has_class_{c}"] == 1].sort_values(f"area_class_{c}", ascending=False),
            f"class {c}, large area",
        ))
    groups.append((index[index.has_defect == 0], "defect-free — false-positive check"))

    empty = [reason for rows, reason in groups if take(rows, reason) == 0]
    if empty:
        raise ValueError(
            "no validation image left for these figure groups: " + "; ".join(empty)
            + ". The figure set is supposed to cover every one of them."
        )
    return chosen


def make_figure(image_id, reason, images_dir, masks_table, models, device, out_path):
    with Image.open(os.path.join(images_dir, image_id)) as img:
        gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    gt = build_mask(masks_table.get(image_id), IMAGE_SHAPE)
    preds = {name: predict(model, gray, device) for name, model in models.items()}

    panels = [
        ("original", np.repeat(gray[:, :, None], 3, axis=2)),
        ("ground truth", overlay(gray, gt)),
        ("U-Net (champion)", overlay(gray, preds["unet"])),
        ("DeepLabV3+ (challenger)", overlay(gray, preds["deeplabv3p"])),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(14, 7.5))
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(panel, aspect="auto")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])

    present = sorted({c + 1 for c in range(4) if gt[c].any()})
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=8,
                          color=CLASS_COLOURS[c], label=f"class {c}")
               for c in (present or [1, 2, 3, 4])]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"{image_id} — {reason}", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--index-csv", default=None)
    ap.add_argument("--n-per-group", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = os.path.join(args.data_dir, "train_images")
    index_csv = args.index_csv or os.path.join(args.data_dir, "index.csv")
    masks_table = load_masks_table(os.path.join(args.data_dir, "train.csv"))
    val_ids = read_split(args.splits_dir, "val")

    models = {name: load_model(name, os.path.join(args.out_dir, "ckpt"), device)
              for name in ("unet", "deeplabv3p")}

    figs_dir = os.path.join(args.out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    chosen = choose_images(index_csv, val_ids, args.n_per_group)
    for i, (image_id, reason) in enumerate(chosen, 1):
        out_path = os.path.join(figs_dir, f"seg_compare_{i:02d}_{image_id.replace('.jpg','')}.png")
        make_figure(image_id, reason, images_dir, masks_table, models, device, out_path)
        print(f"  [{i:2d}] {out_path}  ({reason})")

    print(f"\nwrote {len(chosen)} four-panel figures to {figs_dir}")
    if len(chosen) < 8:
        raise SystemExit(f"phase 4 requires >= 8 figures, only produced {len(chosen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
