"""Segmentation metrics for phase 4, including the empty-mask trap.

The trap. Roughly 47% of this dataset's images carry no defect at all, and inside a
defect-bearing image at least three of the four class channels are usually empty too. The
Kaggle competition's metric scores an (image, class) pair with an empty ground truth and an
empty prediction as a perfect 1.0. Under that rule a model that outputs nothing at all
scores above 0.9 — the number is dominated by correctly-predicted emptiness and says almost
nothing about whether the model can find a defect.

So this module reports three separate things instead of one number:

  1. HEADLINE Dice / mIoU, computed only over (image, class) pairs whose ground truth is
     non-empty. No free credit for emptiness; this is the number that answers "when there is
     a defect, does the model find it".
  2. The false-positive behaviour on the defect-free images, reported separately as a rate.
     A model can only cheat the headline number by predicting more, and this is where that
     shows up, so the pair is honest in a way either half alone is not.
  3. The Kaggle-style all-pairs Dice, kept purely so the report can show the size of the
     inflation rather than assert it.

Everything is derived from one flat table of per-(image, class) records, so the grouped
analyses (per class, small vs large defect) are re-slices of the same evaluation pass rather
than separate runs that could drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairRecord:
    """One (image, class) pair from the validation pass."""
    image_id: str
    class_id: int
    gt_area: int            # ground-truth defect pixels
    pred_area: int          # predicted defect pixels, after post-processing
    intersection: int
    image_has_defect: bool  # true if ANY class is non-empty in this image


def postprocess(probs: np.ndarray, threshold: float = 0.5, min_area: int = 300) -> np.ndarray:
    """Probabilities -> a 0/1 mask, dropping predictions too small to be a real defect.

    Two knobs, both fixed in advance rather than tuned on the validation set — tuning them
    on val and then reporting val numbers would be measuring the tuning, not the model.
    `threshold=0.5` is the neutral choice for a sigmoid. `min_area=300` px is a prior read
    off phase 2's area distribution: genuine defects are far larger than 300 px, so a
    class channel firing on less than that is speckle, and zeroing it is what stops those
    specks from becoming false positives on clean images.
    """
    mask = (probs > threshold).astype(np.uint8)
    for c in range(mask.shape[0]):
        if mask[c].sum() < min_area:
            mask[c] = 0
    return mask


def collect_pairs(image_ids, gt_masks, pred_masks) -> list[PairRecord]:
    """Build the per-pair table for a batch. Masks are (C, H, W) 0/1 arrays."""
    records = []
    for image_id, gt, pred in zip(image_ids, gt_masks, pred_masks):
        has_defect = bool(gt.sum() > 0)
        for c in range(gt.shape[0]):
            g, p = gt[c].astype(bool), pred[c].astype(bool)
            records.append(PairRecord(
                image_id=str(image_id),
                class_id=c + 1,
                gt_area=int(g.sum()),
                pred_area=int(p.sum()),
                intersection=int(np.logical_and(g, p).sum()),
                image_has_defect=has_defect,
            ))
    return records


def _dice(record: PairRecord) -> float:
    denominator = record.gt_area + record.pred_area
    if denominator == 0:
        return 1.0  # both empty — only ever used for the Kaggle-style comparison figure
    return 2.0 * record.intersection / denominator


def _iou(record: PairRecord) -> float:
    union = record.gt_area + record.pred_area - record.intersection
    if union == 0:
        return 1.0
    return record.intersection / union


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarise(records: list[PairRecord], small_large_threshold: int | None = None) -> dict:
    """Turn the pair table into the metric dict written to outputs/metrics/seg_*.json."""
    defect_pairs = [r for r in records if r.gt_area > 0]
    if not defect_pairs:
        raise ValueError("no defect-bearing pairs in the evaluation set")

    # --- 1. headline: defect-bearing pairs only ------------------------------------------
    headline_dice = _mean([_dice(r) for r in defect_pairs])
    headline_miou = _mean([_iou(r) for r in defect_pairs])

    per_class_dice, per_class_iou, per_class_n = {}, {}, {}
    for c in range(1, 5):
        pairs = [r for r in defect_pairs if r.class_id == c]
        per_class_dice[str(c)] = _mean([_dice(r) for r in pairs])
        per_class_iou[str(c)] = _mean([_iou(r) for r in pairs])
        per_class_n[str(c)] = len(pairs)

    # --- 2. false positives on the defect-free images ------------------------------------
    # Grouped by image, because "this clean sheet was flagged" is the decision an inspector
    # actually acts on — a per-pixel rate would be diluted to near zero by the frame size.
    clean_images: dict[str, int] = {}
    for r in records:
        if not r.image_has_defect:
            clean_images[r.image_id] = clean_images.get(r.image_id, 0) + r.pred_area
    n_clean = len(clean_images)
    n_clean_flagged = sum(1 for v in clean_images.values() if v > 0)
    false_positive_rate = (n_clean_flagged / n_clean) if n_clean else None
    mean_fp_area = _mean([float(v) for v in clean_images.values() if v > 0]) or 0.0

    # The same failure inside a defect-bearing image: the right image, the wrong class
    # channel fired. Reported separately because it costs a different thing operationally.
    wrong_class_pairs = [r for r in records if r.image_has_defect and r.gt_area == 0]
    wrong_class_fp_rate = _mean([1.0 if r.pred_area > 0 else 0.0 for r in wrong_class_pairs])

    # --- 3. the inflated number, for contrast --------------------------------------------
    kaggle_style_dice = _mean([_dice(r) for r in records])

    # --- 4. required grouped analysis: small vs large defect ------------------------------
    # Threshold is the median ground-truth area over the defect-bearing pairs, so the two
    # groups are the same size and neither is a handful of outliers. The value is written
    # into the json so the report can quote it instead of hand-waving "small".
    areas = np.array([r.gt_area for r in defect_pairs])
    if small_large_threshold is None:
        small_large_threshold = int(np.median(areas))
    small = [r for r in defect_pairs if r.gt_area <= small_large_threshold]
    large = [r for r in defect_pairs if r.gt_area > small_large_threshold]

    def group(pairs: list[PairRecord]) -> dict:
        return {
            "n_pairs": len(pairs),
            "dice": _mean([_dice(r) for r in pairs]),
            "iou": _mean([_iou(r) for r in pairs]),
            "median_gt_area_px": int(np.median([r.gt_area for r in pairs])) if pairs else None,
            "missed_entirely": sum(1 for r in pairs if r.pred_area == 0),
        }

    return {
        "headline_note": (
            "Dice and mIoU below are computed on defect-bearing (image, class) pairs ONLY. "
            "Pairs with an empty ground truth are excluded because scoring empty-vs-empty "
            "as a perfect 1.0 lets a model that predicts nothing win. The behaviour on "
            "defect-free images is reported separately under false_positives."
        ),
        "headline": {
            "dice_defect_only": headline_dice,
            "miou_defect_only": headline_miou,
            "n_defect_pairs": len(defect_pairs),
        },
        "per_class_dice_defect_only": per_class_dice,
        "per_class_iou_defect_only": per_class_iou,
        "per_class_n_pairs": per_class_n,
        "false_positives": {
            "n_defect_free_images": n_clean,
            "n_flagged": n_clean_flagged,
            "false_positive_rate": false_positive_rate,
            "mean_flagged_area_px": mean_fp_area,
            "wrong_class_fp_rate_within_defect_images": wrong_class_fp_rate,
            "note": (
                "false_positive_rate = share of DEFECT-FREE validation images on which the "
                "model predicted at least one defect pixel after post-processing. It is the "
                "counterweight to the defect-only headline: a model can raise the headline "
                "by predicting more aggressively, and the cost of doing so lands here."
            ),
        },
        "inflated_reference": {
            "kaggle_style_dice_all_pairs": kaggle_style_dice,
            "note": (
                "All (image, class) pairs with empty-vs-empty counted as 1.0, matching the "
                "Kaggle leaderboard convention. Reported only to show how much of that "
                "number is correctly-predicted emptiness; it is NOT the headline."
            ),
        },
        "grouped_by_area": {
            "threshold_px": int(small_large_threshold),
            "threshold_rule": "median ground-truth area over defect-bearing val pairs",
            "small": group(small),
            "large": group(large),
        },
    }


def empty_prediction_baseline(records: list[PairRecord]) -> dict:
    """What the two numbers look like for a model that predicts nothing at all.

    Included in every metrics file as the reference point that makes the trap concrete: the
    Kaggle-style figure stays high while the defect-only Dice goes to 0.
    """
    blanked = [
        PairRecord(r.image_id, r.class_id, r.gt_area, 0, 0, r.image_has_defect)
        for r in records
    ]
    return {
        "dice_defect_only": _mean([_dice(r) for r in blanked if r.gt_area > 0]),
        "kaggle_style_dice_all_pairs": _mean([_dice(r) for r in blanked]),
        "false_positive_rate": 0.0,
    }
