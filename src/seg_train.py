"""Train and evaluate one phase-4 segmentation model.

Both models go through this same file — same split, same loss, same schedule, same seed, same
post-processing — so the only thing differing between the champion and the challenger is the
architecture. Running them through two hand-written loops is how comparisons quietly become
meaningless.

    python src/seg_train.py --model unet        --data-dir /content/steel --epochs 15
    python src/seg_train.py --model deeplabv3p  --data-dir /content/steel --epochs 15
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seg_models  # noqa: E402
from seg_data import make_loaders  # noqa: E402
from seg_losses import DiceBCELoss  # noqa: E402
from seg_metrics import collect_pairs, empty_prediction_baseline, postprocess, summarise  # noqa: E402

SEED = 42


def set_seed(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_context(device: torch.device, enabled: bool = True):
    """Mixed precision on CUDA, a no-op anywhere else.

    `torch.autocast(device_type='mps')` raises outright — it rejects the device before it
    ever looks at `enabled=False` — so the guard has to be around the context manager, not
    inside it. Colab is CUDA, but this keeps a laptop CPU/MPS dry run working.
    """
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return contextlib.nullcontext()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, image_ids, threshold=0.5, min_area=300, amp=True):
    """One pass over the validation loader -> the per-(image, class) record table.

    Predictions are post-processed here, once, so every metric downstream — headline,
    per-class, small/large, false-positive rate — is computed from the identical masks.
    """
    model.eval()
    records = []
    cursor = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        with amp_context(device, amp):
            logits = model(images)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        gts = masks.numpy().astype(np.uint8)

        batch_ids = image_ids[cursor:cursor + len(probs)]
        cursor += len(probs)
        preds = np.stack([postprocess(p, threshold, min_area) for p in probs])
        records.extend(collect_pairs(batch_ids, gts, preds))
    return records


def preflight(model_name: str, batch_size: int = 8, steps: int = 6) -> dict:
    """Time and memory for a few real training steps, before committing to a full run.

    There is one GPU to share, and a 1600x256 frame is unusual enough that guessing whether
    batch 8 fits — or how long an epoch takes — is not worth an hour of finding out. This
    runs the real model on the real frame size with random tensors and reports peak memory
    and a projected epoch time, so the batch size is chosen from a measurement.
    """
    device = pick_device()
    model = seg_models.build(model_name, pretrained=False).to(device)
    criterion = DiceBCELoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    images = torch.randn(batch_size, 3, 256, 1600, device=device)
    masks = torch.zeros(batch_size, 4, 256, 1600, device=device)
    masks[:, 0, 50:100, 200:400] = 1.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    for i in range(steps):
        if i == 1 and device.type == "cuda":
            torch.cuda.synchronize()
            started = time.time()  # skip step 0: it pays the one-off cudnn autotune cost
        optimiser.zero_grad(set_to_none=True)
        with amp_context(device):
            loss = criterion(model(images), masks)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
        per_step = (time.time() - started) / (steps - 1)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        per_step, peak_gb = float("nan"), float("nan")

    n_train = 10054  # phase-0 frozen split
    result = {
        "model": model_name, "batch_size": batch_size, "device": str(device),
        "peak_memory_gb": round(peak_gb, 2), "seconds_per_step": round(per_step, 3),
        "projected_train_minutes_per_epoch": round(per_step * (n_train / batch_size) / 60, 1),
    }
    print(f"preflight {model_name} @ batch {batch_size}: peak {peak_gb:.2f} GB, "
          f"{per_step:.3f} s/step, ~{result['projected_train_minutes_per_epoch']:.1f} min/epoch "
          f"(train only, val adds ~25%)")
    del model, images, masks
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def train_model(
    model_name: str,
    data_dir: str,
    splits_dir: str,
    out_dir: str,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 2,
    limit: int | None = None,
    patience: int = 3,
    threshold: float = 0.5,
    min_area: int = 300,
    resume: bool = True,
) -> dict:
    set_seed()
    device = pick_device()
    print(f"device: {device}")

    train_loader, val_loader, _ = make_loaders(
        data_dir, splits_dir, batch_size=batch_size, num_workers=num_workers, limit=limit,
    )
    val_ids = val_loader.dataset.image_ids
    print(f"train batches: {len(train_loader)} | val images: {len(val_ids)}")

    model = seg_models.build(model_name).to(device)
    n_params = seg_models.count_parameters(model)
    print(f"{model_name}: {n_params/1e6:.1f}M trainable parameters")

    criterion = DiceBCELoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    ckpt_dir = os.path.join(out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_{seg_models.ENCODER}.pth")
    # Written after EVERY epoch, not just improving ones. A Colab runtime is stable for
    # roughly 45 minutes and this run is hours long, so a disconnect is expected rather than
    # exceptional — without this, a drop at epoch 12 of 15 costs the whole run. `best` alone
    # is not enough to resume from: it carries no optimiser or scheduler state, so continuing
    # from it would restart the cosine schedule and re-warm Adam's moments.
    last_path = os.path.join(ckpt_dir, f"{model_name}_{seg_models.ENCODER}_last.pth")
    history_path = os.path.join(out_dir, "metrics", f"seg_{model_name}_history.json")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    history, best_dice, best_epoch, epochs_since_best = [], -1.0, -1, 0
    start_epoch = 1

    if resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["state_dict"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]
        best_dice, best_epoch = state["best_dice"], state["best_epoch"]
        epochs_since_best = state["epochs_since_best"]
        start_epoch = state["epoch"] + 1
        print(f"RESUMED from {last_path}: finished epoch {state['epoch']}, "
              f"best Dice {best_dice:.4f} @ epoch {best_epoch}; continuing at {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        started = time.time()
        running, seen = 0.0, 0
        for step, (images, masks) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with amp_context(device):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            running += loss.item() * len(images)
            seen += len(images)
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running/seen:.4f}", flush=True)
        scheduler.step()

        records = evaluate(model, val_loader, device, val_ids, threshold, min_area)
        metrics = summarise(records)
        val_dice = metrics["headline"]["dice_defect_only"]
        fp_rate = metrics["false_positives"]["false_positive_rate"]
        history.append({
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_dice_defect_only": val_dice,
            "val_miou_defect_only": metrics["headline"]["miou_defect_only"],
            "val_false_positive_rate": fp_rate,
            "seconds": round(time.time() - started, 1),
        })
        print(
            f"epoch {epoch}/{epochs} | train loss {running/max(seen,1):.4f} | "
            f"val Dice(defect-only) {val_dice:.4f} | FP rate {fp_rate:.4f} | "
            f"{time.time()-started:.0f}s",
            flush=True,
        )

        # Selection is on the defect-only Dice, never on the all-pairs figure — selecting on
        # the inflated metric would pick the checkpoint that predicts least.
        stop = False
        if val_dice > best_dice:
            best_dice, best_epoch, epochs_since_best = val_dice, epoch, 0
            torch.save({
                "model": model_name,
                "encoder": seg_models.ENCODER,
                "epoch": epoch,
                "val_dice_defect_only": val_dice,
                "state_dict": model.state_dict(),
            }, ckpt_path)
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                print(f"early stop: no improvement for {patience} epochs")
                stop = True

        # Crash-resume point + the history on disk, every epoch regardless of improvement.
        # Written to a temp file and renamed because os.replace is atomic: if the runtime
        # dies mid-write, the previous checkpoint is still intact rather than truncated.
        torch.save({
            "model": model_name, "encoder": seg_models.ENCODER, "epoch": epoch,
            "state_dict": model.state_dict(), "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "history": history, "best_dice": best_dice, "best_epoch": best_epoch,
            "epochs_since_best": epochs_since_best,
        }, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)
        with open(history_path + ".tmp", "w") as fh:
            json.dump({"model": model_name, "best_epoch": best_epoch,
                       "best_val_dice_defect_only": best_dice, "history": history}, fh, indent=2)
        os.replace(history_path + ".tmp", history_path)

        if stop:
            break

    # Final numbers come from the best checkpoint, not from wherever training happened to end.
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    records = evaluate(model, val_loader, device, val_ids, threshold, min_area)
    metrics = summarise(records)
    metrics.update({
        "model": model_name,
        "role": "champion" if model_name == "unet" else "challenger",
        "encoder": seg_models.ENCODER,
        "encoder_weights": seg_models.ENCODER_WEIGHTS,
        "trainable_parameters": n_params,
        "loss": "0.5 * BCE + 0.5 * (1 - soft Dice)",
        "input_size_hw": [256, 1600],
        "postprocess": {"prob_threshold": threshold, "min_class_area_px": min_area},
        "optimiser": {"name": "AdamW", "lr": lr, "weight_decay": weight_decay,
                      "schedule": "cosine annealing"},
        "batch_size": batch_size,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "seed": SEED,
        "split": {"train_csv": "splits/train.csv", "val_csv": "splits/val.csv",
                  "n_val_images": len(val_ids)},
        "history": history,
        "empty_prediction_baseline": empty_prediction_baseline(records),
        "checkpoint": os.path.relpath(ckpt_path, out_dir),
    })

    metrics_dir = os.path.join(out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"seg_{model_name}.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"wrote {metrics_path}")

    head = metrics["headline"]
    fps = metrics["false_positives"]
    print(f"\nHEADLINE (defect-bearing pairs only): Dice {head['dice_defect_only']:.4f} "
          f"mIoU {head['miou_defect_only']:.4f}")
    print(f"FP rate on {fps['n_defect_free_images']} defect-free images: "
          f"{fps['false_positive_rate']:.4f}")
    print(f"inflated all-pairs Dice for contrast: "
          f"{metrics['inflated_reference']['kaggle_style_dice_all_pairs']:.4f}")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(seg_models.MODELS))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="debug: use only N images")
    args = ap.parse_args()

    train_model(
        args.model, args.data_dir, args.splits_dir, args.out_dir,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        num_workers=args.num_workers, limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
