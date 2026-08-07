"""CPU smoke test for the phase-4 segmentation pipeline.

Runs without the 1.7 GB dataset and without a GPU: it writes a handful of synthetic 1600x256
images and a synthetic train.csv, pushes them through the real Dataset, the real loss, the
real metric code and both real smp models, and checks the answers against values worked out
by hand. The point is to find the silent mistakes — a transposed mask, an augmentation that
flips the image but not the label, a metric that quietly counts empty masks — before a GPU
hour is spent on them.

    python src/test_seg_pipeline.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rle import rle_encode  # noqa: E402
from seg_data import IMAGE_SHAPE, SteelSegDataset, augment, build_mask, load_masks_table, make_loaders  # noqa: E402
from seg_losses import DiceBCELoss, soft_dice_loss  # noqa: E402
from seg_metrics import PairRecord, collect_pairs, empty_prediction_baseline, postprocess, summarise  # noqa: E402

H, W = IMAGE_SHAPE
results: list[bool] = []


def check(name: str, ok: bool) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    results.append(bool(ok))
    return bool(ok)


# ---------------------------------------------------------------------------------------
# 1. mask construction
# ---------------------------------------------------------------------------------------
def test_masks() -> None:
    print("\n[1] RLE -> 4-channel mask")
    truth = np.zeros((4, H, W), dtype=np.uint8)
    truth[0, 10:40, 100:130] = 1     # class 1 block
    truth[2, 200:250, 900:1000] = 1  # class 3 block
    rles = {1: rle_encode(truth[0]), 3: rle_encode(truth[2])}

    mask = build_mask(rles)
    check("channel c-1 carries class c, other channels stay empty",
          np.array_equal(mask, truth))
    check("empty input -> all-zero 4-channel mask (defect-free image)",
          build_mask(None).shape == (4, H, W) and build_mask(None).sum() == 0)

    # Column-major is the thing that fails silently: if the mask came back transposed it
    # would still round-trip, so compare against a mask built by an independent route.
    flat = np.zeros(H * W, dtype=np.uint8)
    flat[0:5] = 1                       # first 5 pixels down column 0
    column_major = flat.reshape((H, W), order="F")
    check("first RLE run fills a COLUMN, not a row",
          np.array_equal(build_mask({2: "1 5"})[1], column_major))


# ---------------------------------------------------------------------------------------
# 2. augmentation keeps image and mask aligned
# ---------------------------------------------------------------------------------------
def test_augment() -> None:
    print("\n[2] augmentation")
    rng = np.random.default_rng(0)
    image = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((4, H, W), dtype=np.uint8)
    image[20:30, 50:60] = 1.0   # a bright patch...
    mask[0, 20:30, 50:60] = 1   # ...labelled in exactly the same place

    misaligned = 0
    for _ in range(30):
        aug_image, aug_mask = augment(image.copy(), mask.copy(), rng)
        # Wherever the label is set, the image must still be the bright patch. A flip
        # applied to one and not the other breaks this immediately.
        where = aug_mask[0].astype(bool)
        if where.sum() != 100 or (aug_image[where] < 0.4).any():
            misaligned += 1
    check("30 random augmentations keep image and mask aligned", misaligned == 0)

    aug_image, _ = augment(image.copy(), mask.copy(), np.random.default_rng(1))
    check("augmented pixels stay in [0, 1]", aug_image.min() >= 0.0 and aug_image.max() <= 1.0)


# ---------------------------------------------------------------------------------------
# 3. loss behaves the way the loss choice claims it does
# ---------------------------------------------------------------------------------------
def test_loss() -> None:
    print("\n[3] Dice + BCE loss")
    targets = torch.zeros(2, 4, 64, 160)
    targets[0, 0, 10:20, 10:30] = 1.0   # ~1% of the frame, like a real defect

    # A confident all-background prediction: BCE is already tiny, Dice is not. This is the
    # data-driven argument for the combined loss, checked rather than asserted in prose.
    all_background = torch.full_like(targets, -8.0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(all_background, targets).item()
    dice = soft_dice_loss(all_background, targets).item()
    check(f"BCE alone barely penalises an empty prediction (BCE={bce:.4f} < 0.05)", bce < 0.05)
    check(f"Dice term does penalise it (dice loss={dice:.4f} > 0.1)", dice > 0.1)

    criterion = DiceBCELoss()
    perfect = torch.where(targets > 0, 8.0, -8.0)
    check("a perfect prediction scores lower than an empty one",
          criterion(perfect, targets).item() < criterion(all_background, targets).item())

    logits = torch.zeros(2, 4, 64, 160, requires_grad=True)
    criterion(logits, targets).backward()
    check("loss is differentiable and produces a finite gradient",
          logits.grad is not None and torch.isfinite(logits.grad).all().item()
          and logits.grad.abs().sum().item() > 0)

    # --- the regression that cost a 40-minute training run --------------------------------
    # Averaging Dice over EMPTY-ground-truth pairs makes the term reward predicting nothing:
    # with no intersection the score is smooth/(pred.sum()+smooth), maximised at pred=0. Since
    # ~85-90% of channel-slots here are empty, that turned the Dice term into a standing
    # instruction to output an empty mask, and the first U-Net run duly collapsed (val Dice
    # 0.58 -> 0.09 while train loss kept falling). These three checks fail on that old
    # behaviour.
    empty = torch.zeros(2, 4, 32, 64)
    pred_something = torch.full_like(empty, 2.0)   # confidently predicts everywhere
    pred_nothing = torch.full_like(empty, -8.0)
    check("all-empty GT: Dice term does NOT prefer the empty prediction",
          abs(soft_dice_loss(pred_something, empty).item()
              - soft_dice_loss(pred_nothing, empty).item()) < 1e-6)

    grad_probe = torch.zeros(2, 4, 32, 64, requires_grad=True)
    soft_dice_loss(grad_probe, empty).backward()
    check("all-empty GT: Dice term contributes no gradient at all",
          grad_probe.grad is None or grad_probe.grad.abs().sum().item() == 0.0)

    # A batch where one pair has a defect and seven are empty: the empty pairs must not
    # dilute the signal from the one that matters.
    mixed = torch.zeros(2, 4, 32, 64)
    mixed[0, 1, 5:15, 5:25] = 1.0
    only = torch.zeros(1, 1, 32, 64)
    only[0, 0, 5:15, 5:25] = 1.0
    logits_mixed = torch.full_like(mixed, -2.0)
    logits_only = torch.full_like(only, -2.0)
    check("empty pairs do not dilute the defect-bearing pair's Dice",
          abs(soft_dice_loss(logits_mixed, mixed).item()
              - soft_dice_loss(logits_only, only).item()) < 1e-5)


# ---------------------------------------------------------------------------------------
# 4. the metric trap
# ---------------------------------------------------------------------------------------
def test_metrics() -> None:
    print("\n[4] metrics — empty-mask trap, FP rate, area grouping")
    # 4 defect-bearing images (one non-empty class each) and 6 defect-free ones.
    records: list[PairRecord] = []
    for i, area in enumerate([100, 200, 5000, 9000]):
        for c in range(1, 5):
            is_gt = (c == 1)
            records.append(PairRecord(f"d{i}.jpg", c, area if is_gt else 0, 0, 0, True))
    for i in range(6):
        for c in range(1, 5):
            records.append(PairRecord(f"c{i}.jpg", c, 0, 0, 0, False))

    blank = summarise(records)
    check("a model predicting NOTHING gets Dice 0.0 on the defect-only headline",
          abs(blank["headline"]["dice_defect_only"] - 0.0) < 1e-9)
    # 40 pairs, 36 of them empty-vs-empty -> 36/40 = 0.900 handed out for predicting nothing.
    check(f"...while the Kaggle-style all-pairs number rewards it with "
          f"{blank['inflated_reference']['kaggle_style_dice_all_pairs']:.3f}",
          abs(blank["inflated_reference"]["kaggle_style_dice_all_pairs"] - 0.9) < 1e-9)
    check("its false-positive rate is 0 — which is exactly why FP must be reported too",
          blank["false_positives"]["false_positive_rate"] == 0.0)
    check("defect-free images are counted by image, not by pair (6, not 24)",
          blank["false_positives"]["n_defect_free_images"] == 6)

    # Now flag 3 of the 6 clean images -> FP rate must be 0.5.
    flagged = [
        PairRecord(r.image_id, r.class_id, r.gt_area,
                   400 if (not r.image_has_defect and r.image_id in {"c0.jpg", "c1.jpg", "c2.jpg"}
                           and r.class_id == 3) else 0,
                   0, r.image_has_defect)
        for r in records
    ]
    check("3 of 6 clean images flagged -> FP rate 0.500",
          abs(summarise(flagged)["false_positives"]["false_positive_rate"] - 0.5) < 1e-9)

    # Perfect predictions -> Dice 1.0, and the small/large split lands where it should.
    perfect = [PairRecord(r.image_id, r.class_id, r.gt_area, r.gt_area, r.gt_area,
                          r.image_has_defect) for r in records]
    got = summarise(perfect)
    check("a perfect model scores Dice 1.0 on the headline",
          abs(got["headline"]["dice_defect_only"] - 1.0) < 1e-9)
    grouped = got["grouped_by_area"]
    check(f"small/large threshold is the median area (got {grouped['threshold_px']}, "
          f"expected 2600)", grouped["threshold_px"] == 2600)
    check("small and large groups split the 4 defect pairs 2/2",
          grouped["small"]["n_pairs"] == 2 and grouped["large"]["n_pairs"] == 2)

    baseline = empty_prediction_baseline(perfect)
    # Fed PERFECT predictions, the baseline must still report what an EMPTY model would get,
    # i.e. it blanks the predictions itself rather than reading them off the input.
    check("empty-prediction baseline is recomputed from the GT, not copied from the input",
          baseline["dice_defect_only"] == 0.0
          and abs(baseline["kaggle_style_dice_all_pairs"] - 0.9) < 1e-9)

    # post-processing
    probs = np.zeros((4, 64, 160), dtype=np.float32)
    probs[0, 0:10, 0:50] = 0.9   # 500 px — a real defect
    probs[1, 0:2, 0:5] = 0.9     # 10 px — speckle
    out = postprocess(probs, threshold=0.5, min_area=300)
    check("post-processing keeps the 500 px region and drops the 10 px speckle",
          out[0].sum() == 500 and out[1].sum() == 0)

    pairs = collect_pairs(["x.jpg"], [np.stack([out[0]] * 4)], [out])
    check("collect_pairs computes intersection correctly",
          pairs[0].gt_area == 500 and pairs[0].pred_area == 500 and pairs[0].intersection == 500)
    check("collect_pairs marks the image as defect-bearing from the GT, not the prediction",
          all(p.image_has_defect for p in pairs))


# ---------------------------------------------------------------------------------------
# 5. Dataset / DataLoader end to end on synthetic files
# ---------------------------------------------------------------------------------------
def test_dataset(tmp: str) -> None:
    print("\n[5] Dataset + DataLoader on synthetic images")
    data_dir = os.path.join(tmp, "data")
    images_dir = os.path.join(data_dir, "train_images")
    splits_dir = os.path.join(tmp, "splits")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    rng = np.random.default_rng(0)
    rows, ids = [], []
    for i in range(8):
        image_id = f"synth{i:02d}.jpg"
        ids.append(image_id)
        arr = rng.integers(60, 190, size=(H, W), dtype=np.uint8)
        if i % 2 == 0:  # half defect-bearing, half clean — mirrors the real 53/47 balance
            mask = np.zeros((H, W), dtype=np.uint8)
            mask[50:90, 200 + 30 * i:260 + 30 * i] = 1
            arr[mask.astype(bool)] = 240
            rows.append({"ImageId": image_id, "ClassId": (i % 4) + 1,
                         "EncodedPixels": rle_encode(mask)})
        Image.fromarray(arr, mode="L").save(os.path.join(images_dir, image_id))

    pd.DataFrame(rows).to_csv(os.path.join(data_dir, "train.csv"), index=False)
    pd.DataFrame({"image_id": ids[:6], "split": "train"}).to_csv(
        os.path.join(splits_dir, "train.csv"), index=False)
    pd.DataFrame({"image_id": ids[6:], "split": "val"}).to_csv(
        os.path.join(splits_dir, "val.csv"), index=False)

    table = load_masks_table(os.path.join(data_dir, "train.csv"))
    check("load_masks_table skips defect-free images (4 of 8 have entries)", len(table) == 4)

    dataset = SteelSegDataset(ids, images_dir, table, train=False)
    image, mask = dataset[0]
    check("dataset returns image (3,256,1600) float32", tuple(image.shape) == (3, H, W)
          and image.dtype == torch.float32)
    check("dataset returns mask (4,256,1600) float32", tuple(mask.shape) == (4, H, W)
          and mask.dtype == torch.float32)
    check("grayscale is replicated across all 3 channels",
          torch.equal(image[0], image[1]) and torch.equal(image[1], image[2]))
    check("mask is 0/1 only", set(torch.unique(mask).tolist()) <= {0.0, 1.0})
    check("defect-bearing sample has exactly one non-empty class channel",
          int((mask.sum(dim=(1, 2)) > 0).sum()) == 1)
    _, clean_mask = dataset[1]
    check("defect-free sample has an all-zero mask", clean_mask.sum().item() == 0)

    train_loader, val_loader, _ = make_loaders(data_dir, splits_dir, batch_size=2, num_workers=0)
    images, masks = next(iter(train_loader))
    check("train loader batches to (2,3,256,1600) / (2,4,256,1600)",
          tuple(images.shape) == (2, 3, H, W) and tuple(masks.shape) == (2, 4, H, W))
    check("val loader reads the frozen split, in file order",
          val_loader.dataset.image_ids == ids[6:])


# ---------------------------------------------------------------------------------------
# 6. both real smp models, forward + backward on CPU
# ---------------------------------------------------------------------------------------
def test_models() -> None:
    print("\n[6] U-Net and DeepLabV3+ forward/backward (CPU, tiny input)")
    import seg_models

    # 64x160 rather than the real 256x1600: same /32-divisible property, ~40x less compute,
    # so this runs on a laptop in seconds. Encoder weights are off — the smoke test must not
    # depend on downloading ImageNet checkpoints.
    x = torch.randn(2, 3, 64, 160)
    y = torch.zeros(2, 4, 64, 160)
    y[0, 2, 10:30, 20:60] = 1.0
    criterion = DiceBCELoss()

    for name in ("unet", "deeplabv3p"):
        model = seg_models.build(name, pretrained=False)
        model.train()
        logits = model(x)
        check(f"{name} outputs 4 channels at input resolution",
              tuple(logits.shape) == (2, 4, 64, 160))
        loss = criterion(logits, y)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        check(f"{name} backward produces finite gradients through {len(grads)} tensors",
              len(grads) > 0 and all(torch.isfinite(g).all() for g in grads))
        check(f"{name} loss is finite ({loss.item():.4f})", np.isfinite(loss.item()))

    check("both models use the same encoder, so the comparison is architecture-only",
          seg_models.ENCODER == "resnet34")

    # A real 1600x256 frame must pass through without a shape error — the native resolution
    # is the whole reason no resizing step exists.
    model = seg_models.build("unet", pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, H, W))
    check("U-Net accepts the native 256x1600 frame unresized",
          tuple(out.shape) == (1, 4, H, W))


# ---------------------------------------------------------------------------------------
# 7. figure pipeline — the >= 8 four-panel comparison figures
# ---------------------------------------------------------------------------------------
def test_figures(tmp: str) -> None:
    print("\n[7] four-panel figure pipeline")
    import seg_figs
    import seg_models

    data_dir = os.path.join(tmp, "data")
    images_dir = os.path.join(data_dir, "train_images")
    splits_dir = os.path.join(tmp, "splits")
    figs_dir = os.path.join(tmp, "outputs", "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # A synthetic index.csv shaped exactly like build_index.py's output, covering every
    # selection group the figure chooser asks for.
    rng = np.random.default_rng(1)
    rows, ids = [], []
    csv_rows = []
    for i in range(20):
        image_id = f"fig{i:02d}.jpg"
        ids.append(image_id)
        arr = rng.integers(60, 190, size=(H, W), dtype=np.uint8)
        areas = {c: 0 for c in range(1, 5)}
        if i < 16:
            primary = (i % 4) + 1
            width = 10 + 20 * (i // 4)
            mask = np.zeros((H, W), dtype=np.uint8)
            mask[60:60 + width, 300:300 + width * 3] = 1
            arr[mask.astype(bool)] = 235
            areas[primary] = int(mask.sum())
            csv_rows.append({"ImageId": image_id, "ClassId": primary,
                             "EncodedPixels": rle_encode(mask)})
            if i == 15:  # one multi-class image
                second = np.zeros((H, W), dtype=np.uint8)
                second[150:180, 900:1000] = 1
                other = 1 if primary != 1 else 2
                areas[other] = int(second.sum())
                csv_rows.append({"ImageId": image_id, "ClassId": other,
                                 "EncodedPixels": rle_encode(second)})
        Image.fromarray(arr, mode="L").save(os.path.join(images_dir, image_id))
        n_classes = sum(1 for v in areas.values() if v > 0)
        rows.append({
            "image_id": image_id, "has_defect": int(n_classes > 0),
            "n_defect_classes": n_classes,
            "class_ids": "|".join(str(c) for c, v in areas.items() if v > 0),
            "primary_class": max(areas, key=areas.get) if n_classes else 0,
            "defect_area_px": sum(areas.values()),
            **{f"area_class_{c}": areas[c] for c in range(1, 5)},
            **{f"has_class_{c}": int(areas[c] > 0) for c in range(1, 5)},
        })
    index_csv = os.path.join(data_dir, "index.csv")
    pd.DataFrame(rows).to_csv(index_csv, index=False)
    pd.concat([pd.read_csv(os.path.join(data_dir, "train.csv")),
               pd.DataFrame(csv_rows)]).to_csv(os.path.join(data_dir, "train.csv"), index=False)
    pd.DataFrame({"image_id": ids, "split": "val"}).to_csv(
        os.path.join(splits_dir, "val.csv"), index=False)

    chosen = seg_figs.choose_images(index_csv, ids, n_per_group=2)
    check(f"chooser returns >= 8 images (got {len(chosen)})", len(chosen) >= 8)
    check("chooser returns no duplicates", len({c[0] for c in chosen}) == len(chosen))
    reasons = {c[1] for c in chosen}
    check("selection covers all 4 classes, the smallest defects, multi-class and clean",
          sum(f"class {c}, large area" in reasons for c in range(1, 5)) == 4
          and any("smallest" in r for r in reasons)
          and any("multiple" in r for r in reasons)
          and any("defect-free" in r for r in reasons))

    # Untrained models: the figures will be noise, but rendering must not depend on that.
    device = torch.device("cpu")
    models = {name: seg_models.build(name, pretrained=False).eval() for name in ("unet", "deeplabv3p")}
    table = load_masks_table(os.path.join(data_dir, "train.csv"))
    image_id, reason = chosen[0]
    out_path = os.path.join(figs_dir, "smoke_panel.png")
    seg_figs.make_figure(image_id, reason, images_dir, table, models, device, out_path)
    check("make_figure writes a non-empty 4-panel png",
          os.path.exists(out_path) and os.path.getsize(out_path) > 10_000)

    gray = np.linspace(0, 1, H * W, dtype=np.float32).reshape(H, W)
    mask = np.zeros((4, H, W), dtype=np.uint8)
    mask[0, 10:20, 10:20] = 1
    tinted = seg_figs.overlay(gray, mask)
    check("overlay tints only the masked pixels and leaves the rest grayscale",
          not np.allclose(tinted[15, 15], gray[15, 15])
          and np.allclose(tinted[200, 200], gray[200, 200]))


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="seg_smoke_")
    try:
        test_masks()
        test_augment()
        test_loss()
        test_metrics()
        test_dataset(tmp)
        test_models()
        test_figures(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
