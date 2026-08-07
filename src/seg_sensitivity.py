"""Operating-point sensitivity: what the probability threshold and min-area filter buy and cost.

The headline metrics are reported at a single pre-registered operating point (probability 0.5,
minimum 300 px per class channel), chosen before any model was trained so the numbers measure
the model rather than a search over post-processing. But 0.5 is a convention, not a
requirement, and on this problem the choice is consequential: the same weights score very
differently depending on how aggressively their probabilities are thresholded.

That matters beyond the leaderboard. On a production line the two errors have different prices
— a missed defect ships a flawed coil, a false alarm costs a re-inspection — and the ratio is a
business fact, not a modelling one. So the honest deliverable is the curve plus a named
recommended point, not a single number that silently embeds one cost assumption.

Sweeping this on the validation set and then quoting the best cell as the headline would be
measuring the sweep. The headline therefore stays at the fixed prior; this is analysis
alongside it, clearly labelled.

    python src/seg_sensitivity.py --data-dir /content/steel --out-dir outputs
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seg_models  # noqa: E402
from seg_data import make_loaders  # noqa: E402
from seg_metrics import collect_pairs, postprocess, summarise  # noqa: E402
from seg_train import amp_context, pick_device  # noqa: E402

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MIN_AREAS = [0, 300, 1000, 2000, 4000]
FIXED = (0.5, 300)  # the pre-registered headline point


@torch.no_grad()
def cache_probabilities(model, loader, device, image_ids):
    """Run the model ONCE and keep the per-pair statistics needed for every operating point.

    Re-running inference for each of the 30 combinations would take 30x the GPU time for
    identical predictions. Storing full probability maps for 2,514 images at 4x256x1600 would
    need ~40 GB, so instead each image's probabilities are thresholded at every candidate
    value as it passes through, and only the resulting pair statistics are kept.
    """
    model.eval()
    per_point: dict[tuple[float, int], list] = {(t, a): [] for t in THRESHOLDS for a in MIN_AREAS}
    cursor = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        with amp_context(device):
            logits = model(images)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        gts = masks.numpy().astype(np.uint8)
        batch_ids = image_ids[cursor:cursor + len(probs)]
        cursor += len(probs)
        for threshold in THRESHOLDS:
            for min_area in MIN_AREAS:
                preds = np.stack([postprocess(p, threshold, min_area) for p in probs])
                per_point[(threshold, min_area)].extend(collect_pairs(batch_ids, gts, preds))
    return per_point


def sweep(model_name: str, data_dir: str, splits_dir: str, out_dir: str,
          batch_size: int = 8, num_workers: int = 4) -> dict:
    device = pick_device()
    _, val_loader, _ = make_loaders(data_dir, splits_dir, batch_size=batch_size,
                                    num_workers=num_workers)
    val_ids = val_loader.dataset.image_ids

    ckpt = os.path.join(out_dir, "ckpt", f"{model_name}_{seg_models.ENCODER}.pth")
    model = seg_models.build(model_name, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])

    per_point = cache_probabilities(model, val_loader, device, val_ids)

    rows = []
    for (threshold, min_area), records in sorted(per_point.items()):
        m = summarise(records)
        rows.append({
            "prob_threshold": threshold,
            "min_class_area_px": min_area,
            "dice_defect_only": m["headline"]["dice_defect_only"],
            "miou_defect_only": m["headline"]["miou_defect_only"],
            "false_positive_rate": m["false_positives"]["false_positive_rate"],
            "kaggle_style_dice_all_pairs": m["inflated_reference"]["kaggle_style_dice_all_pairs"],
            "is_headline_point": (threshold, min_area) == FIXED,
        })

    # A recommendation needs a stated rule, or it is just the cell that flattered the model.
    # Rule: among points holding at least 95% of the headline Dice, take the lowest FP rate —
    # i.e. buy the largest reduction in false alarms that costs almost no detection quality.
    headline = next(r for r in rows if r["is_headline_point"])
    floor = 0.95 * headline["dice_defect_only"]
    eligible = [r for r in rows if r["dice_defect_only"] >= floor]
    recommended = min(eligible, key=lambda r: r["false_positive_rate"]) if eligible else headline

    return {
        "model": model_name,
        "note": (
            "The headline metrics elsewhere are reported at the pre-registered operating point "
            "(threshold 0.5, min area 300 px), fixed before training so they measure the model "
            "and not a search over post-processing. This sweep is analysis alongside that "
            "number, never a replacement for it."
        ),
        "headline_point": {"prob_threshold": FIXED[0], "min_class_area_px": FIXED[1]},
        "recommendation_rule": (
            "lowest false-positive rate among operating points retaining >= 95% of the "
            "headline defect-only Dice"
        ),
        "recommended_point": recommended,
        "grid": rows,
    }


def plot(results: list[dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5.2), squeeze=False)
    for ax, res in zip(axes[0], results):
        for min_area in MIN_AREAS:
            pts = sorted([r for r in res["grid"] if r["min_class_area_px"] == min_area],
                         key=lambda r: r["false_positive_rate"])
            ax.plot([r["false_positive_rate"] for r in pts],
                    [r["dice_defect_only"] for r in pts],
                    marker="o", markersize=4, label=f"min area {min_area} px")
        head = next(r for r in res["grid"] if r["is_headline_point"])
        ax.scatter([head["false_positive_rate"]], [head["dice_defect_only"]],
                   s=170, facecolors="none", edgecolors="black", linewidths=2,
                   label="headline point (0.5 / 300 px)", zorder=5)
        rec = res["recommended_point"]
        ax.scatter([rec["false_positive_rate"]], [rec["dice_defect_only"]],
                   s=170, marker="*", color="crimson",
                   label="recommended operating point", zorder=6)
        ax.set_xlabel("false-positive rate on defect-free images  (lower is better)")
        ax.set_ylabel("Dice on defect-bearing pairs  (higher is better)")
        ax.set_title(f"{res['model']} — operating-point trade-off")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--models", nargs="+", default=["unet", "deeplabv3p"])
    args = ap.parse_args()

    results = []
    for name in args.models:
        print(f"sweeping {name} over {len(THRESHOLDS)}x{len(MIN_AREAS)} operating points ...")
        res = sweep(name, args.data_dir, args.splits_dir, args.out_dir)
        results.append(res)
        head = next(r for r in res["grid"] if r["is_headline_point"])
        rec = res["recommended_point"]
        print(f"  headline   thr {head['prob_threshold']} area {head['min_class_area_px']:5d}"
              f" -> Dice {head['dice_defect_only']:.4f}  FP {head['false_positive_rate']:.4f}")
        print(f"  recommended thr {rec['prob_threshold']} area {rec['min_class_area_px']:5d}"
              f" -> Dice {rec['dice_defect_only']:.4f}  FP {rec['false_positive_rate']:.4f}")

    metrics_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "seg_sensitivity.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {metrics_dir}/seg_sensitivity.json")

    figs_dir = os.path.join(args.out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    plot(results, os.path.join(figs_dir, "seg_operating_point_sensitivity.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
