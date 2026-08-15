"""Leakage-safe utilities for the steel-defect classification experiments.

This module adds the three high-leverage changes without depending on notebook
state: full-width or tiled inference, horizontal-flip TTA, and per-class
threshold calibration using an inner calibration split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ModuleNotFoundError:
    average_precision_score = None
    roc_auc_score = None

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # Pure calibration/split checks can run without PyTorch.
    torch = None
    F = None


N_CLASSES = 4
DEFAULT_THRESHOLDS = np.full(N_CLASSES, 0.5, dtype=np.float64)


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for model inference; run this part in the Colab GPU environment"
        )


def _inference_mode():
    if torch is not None:
        return torch.inference_mode()

    def decorator(function):
        def unavailable(*args, **kwargs):
            _require_torch()

        return unavailable

    return decorator


def horizontal_tile_starts(width: int, tile_width: int, overlap: int) -> list[int]:
    """Return deterministic tile starts that cover the full image width."""
    if width <= 0 or tile_width <= 0:
        raise ValueError("width and tile_width must be positive")
    if tile_width > width:
        raise ValueError("tile_width cannot exceed image width")
    if overlap < 0 or overlap >= tile_width:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_width")
    if tile_width == width:
        return [0]

    step = tile_width - overlap
    starts = list(range(0, width - tile_width + 1, step))
    last = width - tile_width
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def horizontal_tiles(
    images: torch.Tensor, tile_width: int = 896, overlap: int = 192
) -> list[torch.Tensor]:
    """Split a BCHW tensor into overlapping width tiles."""
    _require_torch()
    if images.ndim != 4:
        raise ValueError(f"images must be BCHW, got shape {tuple(images.shape)}")
    starts = horizontal_tile_starts(images.shape[-1], tile_width, overlap)
    return [images[..., start : start + tile_width] for start in starts]


def _views(
    images: torch.Tensor,
    mode: str,
    resized_width: int,
    tile_width: int,
    overlap: int,
) -> list[torch.Tensor]:
    _require_torch()
    if mode == "resize_800":
        return [
            F.interpolate(
                images,
                size=(images.shape[-2], resized_width),
                mode="bilinear",
                align_corners=False,
            )
        ]
    if mode == "full_width":
        return [images]
    if mode == "tiles":
        return horizontal_tiles(images, tile_width=tile_width, overlap=overlap)
    raise ValueError("mode must be resize_800, full_width, or tiles")


def forward_training_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
    mode: str = "full_width",
    resized_width: int = 800,
    tile_width: int = 896,
    overlap: int = 192,
) -> torch.Tensor:
    """Differentiable bag-level forward pass for training.

    Tiled mode uses max-logit multiple-instance pooling: an image-level class is
    positive when at least one tile contains evidence for it. This avoids giving
    every tile the full-image label as an independent training example.
    """
    _require_torch()
    views = _views(images, mode, resized_width, tile_width, overlap)
    logits = torch.stack([model(view) for view in views], dim=1)
    if logits.shape[-1] != N_CLASSES:
        raise ValueError(f"model must emit {N_CLASSES} logits, got {logits.shape[-1]}")
    return logits.amax(dim=1) if len(views) > 1 else logits[:, 0]


@_inference_mode()
def predict_probabilities(
    model: torch.nn.Module,
    images: torch.Tensor,
    mode: str = "full_width",
    resized_width: int = 800,
    tile_width: int = 896,
    overlap: int = 192,
    horizontal_flip_tta: bool = True,
    tile_pool: str = "max",
) -> torch.Tensor:
    """Return Bx4 probabilities with optional flip TTA and tile aggregation."""
    _require_torch()
    views = _views(images, mode, resized_width, tile_width, overlap)
    per_view = []
    for view in views:
        probs = torch.sigmoid(model(view))
        if horizontal_flip_tta:
            flipped = torch.flip(view, dims=(-1,))
            probs = 0.5 * (probs + torch.sigmoid(model(flipped)))
        per_view.append(probs)

    stacked = torch.stack(per_view, dim=1)
    if stacked.shape[-1] != N_CLASSES:
        raise ValueError(f"model must emit {N_CLASSES} logits, got {stacked.shape[-1]}")
    if len(per_view) == 1:
        return stacked[:, 0]
    if tile_pool == "max":
        return stacked.amax(dim=1)
    if tile_pool == "mean":
        return stacked.mean(dim=1)
    raise ValueError("tile_pool must be max or mean")


def _validate_arrays(labels: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=np.float64)
    if labels.shape != probs.shape or labels.ndim != 2 or labels.shape[1] != N_CLASSES:
        raise ValueError(
            f"labels and probs must both be Nx{N_CLASSES}; got {labels.shape} and {probs.shape}"
        )
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("probs must be finite values in [0, 1]")
    return labels.astype(np.uint8), probs


def _binary_counts(labels: np.ndarray, pred: np.ndarray) -> tuple[int, int, int]:
    tp = int(np.logical_and(labels == 1, pred == 1).sum())
    fp = int(np.logical_and(labels == 0, pred == 1).sum())
    fn = int(np.logical_and(labels == 1, pred == 0).sum())
    return tp, fp, fn


def _binary_f1(labels: np.ndarray, pred: np.ndarray) -> float:
    tp, fp, fn = _binary_counts(labels, pred)
    denominator = 2 * tp + fp + fn
    return float((2 * tp) / denominator) if denominator else 0.0


def calibrate_thresholds(
    labels: np.ndarray,
    probs: np.ndarray,
    grid: Iterable[float] | None = None,
) -> dict:
    """Select one F1-maximising threshold per class on inner calibration data."""
    labels, probs = _validate_arrays(labels, probs)
    candidates = np.asarray(
        list(grid) if grid is not None else np.linspace(0.05, 0.95, 91),
        dtype=np.float64,
    )
    if candidates.ndim != 1 or len(candidates) == 0:
        raise ValueError("threshold grid must be a non-empty one-dimensional sequence")

    thresholds: list[float] = []
    details: dict[str, dict] = {}
    for class_index in range(N_CLASSES):
        y = labels[:, class_index]
        if y.sum() == 0:
            raise ValueError(
                f"class {class_index + 1} has no positives in inner calibration; "
                "regenerate a better stratified inner split"
            )
        scored = []
        for threshold in candidates:
            pred = (probs[:, class_index] >= threshold).astype(np.uint8)
            score = _binary_f1(y, pred)
            scored.append((float(score), float(threshold), int(pred.sum())))
        best_score = max(row[0] for row in scored)
        tied = [row for row in scored if np.isclose(row[0], best_score, atol=1e-12)]
        # Conservative deterministic tie-break: stay closest to the neutral 0.5.
        best = min(tied, key=lambda row: (abs(row[1] - 0.5), -row[1]))
        thresholds.append(best[1])
        details[str(class_index + 1)] = {
            "threshold": best[1],
            "inner_f1": best[0],
            "n_positive_labels": int(y.sum()),
            "n_positive_predictions": best[2],
        }
    return {"thresholds": thresholds, "per_class": details}


def apply_thresholds(probs: np.ndarray, thresholds: Iterable[float]) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    thresholds = np.asarray(list(thresholds), dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != N_CLASSES:
        raise ValueError(f"probs must be Nx{N_CLASSES}")
    if thresholds.shape != (N_CLASSES,):
        raise ValueError(f"thresholds must contain {N_CLASSES} values")
    return (probs >= thresholds[None, :]).astype(np.uint8)


def evaluate_multilabel(
    labels: np.ndarray, probs: np.ndarray, thresholds: Iterable[float]
) -> dict:
    labels, probs = _validate_arrays(labels, probs)
    thresholds = np.asarray(list(thresholds), dtype=np.float64)
    pred = apply_thresholds(probs, thresholds)

    per_class = {}
    f1_values = []
    total_tp = total_fp = total_fn = 0
    for class_index in range(N_CLASSES):
        y = labels[:, class_index]
        p = probs[:, class_index]
        class_pred = pred[:, class_index]
        class_f1 = _binary_f1(y, class_pred)
        f1_values.append(class_f1)
        tp, fp, fn = _binary_counts(y, class_pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        row = {
            "f1": class_f1,
            "accuracy": float((y == class_pred).mean()),
            "n_positive_labels": int(y.sum()),
            "n_positive_predictions": int(class_pred.sum()),
        }
        if len(np.unique(y)) == 2 and roc_auc_score is not None:
            row["roc_auc"] = float(roc_auc_score(y, p))
            row["pr_auc"] = float(average_precision_score(y, p))
        per_class[str(class_index + 1)] = row

    micro_denominator = 2 * total_tp + total_fp + total_fn
    micro_f1 = (2 * total_tp / micro_denominator) if micro_denominator else 0.0

    return {
        "thresholds": thresholds.tolist(),
        "n_images": int(len(labels)),
        "subset_exact_match": float(np.all(labels == pred, axis=1).mean()),
        "macro_f1": float(np.mean(f1_values)),
        "micro_f1": float(micro_f1),
        "per_class": per_class,
    }


def save_calibration(result: dict, output_path: str, split_name: str) -> None:
    if split_name != "inner_calibration":
        raise ValueError(
            "threshold calibration is allowed only on inner_calibration; "
            "never tune on the frozen report validation set"
        )
    payload = {
        "source_split": split_name,
        "selection_metric": "per-class F1",
        "grid": "0.05..0.95 inclusive, step 0.01",
        **result,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_thresholds(path: str) -> list[float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, list) or len(thresholds) != N_CLASSES:
        raise ValueError(f"{path} does not contain four thresholds")
    return [float(value) for value in thresholds]


def _calibrate_cli(args: argparse.Namespace) -> int:
    labels = np.load(args.labels)
    probs = np.load(args.probs)
    result = calibrate_thresholds(labels, probs)
    save_calibration(result, args.out, args.split_name)
    print(json.dumps(result, indent=2))
    return 0


def _evaluate_cli(args: argparse.Namespace) -> int:
    labels = np.load(args.labels)
    probs = np.load(args.probs)
    result = evaluate_multilabel(labels, probs, _load_thresholds(args.thresholds))
    payload = {"split": args.split_name, **result}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--labels", required=True)
    calibrate.add_argument("--probs", required=True)
    calibrate.add_argument("--out", default="outputs/metrics/cls_thresholds_inner.json")
    calibrate.add_argument("--split-name", default="inner_calibration")
    calibrate.set_defaults(handler=_calibrate_cli)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--labels", required=True)
    evaluate.add_argument("--probs", required=True)
    evaluate.add_argument("--thresholds", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--split-name", default="outer_val")
    evaluate.set_defaults(handler=_evaluate_cli)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
