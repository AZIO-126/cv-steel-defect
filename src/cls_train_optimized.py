"""Train leakage-safe classification candidates on the Severstal data.

Run candidate experiments on inner_train/inner_calibration only. The script
saves probabilities and labels so ``cls_optimization.py calibrate`` can freeze
four class-specific thresholds before the one-shot outer validation report.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    EfficientNet_B2_Weights,
    ResNet50_Weights,
    efficientnet_b2,
    resnet50,
)

from cls_optimization import (
    DEFAULT_THRESHOLDS,
    apply_thresholds,
    evaluate_multilabel,
    forward_training_logits,
    predict_probabilities,
)


SEED = 42
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 1600
LABEL_COLUMNS = [f"has_class_{c}" for c in range(1, 5)]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SteelClassificationDataset(Dataset):
    def __init__(
        self,
        index_path: str,
        split_path: str,
        images_dir: str,
        augment: bool,
        seed: int = SEED,
    ) -> None:
        index = pd.read_csv(index_path)
        split = pd.read_csv(split_path)
        required = {"image_id", *LABEL_COLUMNS}
        missing = required - set(index.columns)
        if missing:
            raise ValueError(f"index is missing columns: {sorted(missing)}")
        if "image_id" not in split.columns:
            raise ValueError(f"{split_path} must contain image_id")

        index["image_id"] = index["image_id"].astype(str)
        split["image_id"] = split["image_id"].astype(str)
        merged = split[["image_id"]].merge(
            index[["image_id", *LABEL_COLUMNS]], on="image_id", how="left", validate="one_to_one"
        )
        if merged[LABEL_COLUMNS].isna().any().any():
            raise ValueError("some split image IDs are absent from data/index.csv")
        self.frame = merged
        self.images_dir = images_dir
        self.augment_enabled = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.frame)

    def _augment(self, image: Image.Image, index: int) -> Image.Image:
        if not self.augment_enabled:
            return image
        # Worker-independent deterministic stream for reproducible reruns.
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        if rng.random() < 0.5:
            image = ImageOps.mirror(image)
        if rng.random() < 0.5:
            image = ImageOps.flip(image)
        if rng.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.85, 1.15)))
        if rng.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.85, 1.15)))
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        image_id = str(row["image_id"])
        path = os.path.join(self.images_dir, image_id)
        with Image.open(path) as source:
            image = source.convert("L")
            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
            image = self._augment(image, index)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        target = torch.tensor(row[LABEL_COLUMNS].to_numpy(dtype=np.float32))
        return tensor, target, image_id


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    if arch == "effnetb2":
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = efficientnet_b2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
        return model
    if arch == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 4)
        return model
    raise ValueError("arch must be effnetb2 or resnet50")


def make_loader(
    index_path: str,
    split_path: str,
    images_dir: str,
    batch_size: int,
    train: bool,
    workers: int,
) -> DataLoader:
    dataset = SteelClassificationDataset(
        index_path=index_path,
        split_path=split_path,
        images_dir=images_dir,
        augment=train,
    )
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    tile_width: int,
    overlap: int,
    use_tta: bool,
) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_probs, all_labels, all_ids = [], [], []
    for images, labels, image_ids in loader:
        images = images.to(device, non_blocking=True)
        probs = predict_probabilities(
            model,
            images,
            mode=mode,
            tile_width=tile_width,
            overlap=overlap,
            horizontal_flip_tta=use_tta,
        )
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_ids.extend(image_ids)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(np.uint8)
    pred = apply_thresholds(probs, DEFAULT_THRESHOLDS)
    macro_f1 = f1_score(labels, pred, average="macro", zero_division=0)
    return float(macro_f1), probs, labels, all_ids


def train(args: argparse.Namespace) -> dict:
    if args.final_fit:
        if not args.allow_outer_val:
            raise ValueError("--final-fit requires --allow-outer-val")
        if not args.thresholds:
            raise ValueError("--final-fit requires thresholds frozen on inner_calibration")
    elif not args.allow_outer_val and "inner" not in Path(args.eval_split).name.lower():
        raise ValueError(
            "candidate selection must use inner_calibration.csv. "
            "Pass --allow-outer-val only for the final one-shot report run."
        )
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} arch={args.arch} mode={args.mode} tta={args.tta}")

    train_loader = make_loader(
        args.index, args.train_split, args.images_dir, args.batch_size, True, args.workers
    )
    eval_loader = make_loader(
        args.index, args.eval_split, args.images_dir, args.batch_size, False, args.workers
    )
    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    output = Path(args.output_dir)
    (output / "ckpt").mkdir(parents=True, exist_ok=True)
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    checkpoint = output / "ckpt" / f"{args.tag}.pth"
    best_f1, best_epoch, epochs_without_gain = -1.0, -1, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        total_loss, n_rows = 0.0, 0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = forward_training_logits(
                    model,
                    images,
                    mode=args.mode,
                    tile_width=args.tile_width,
                    overlap=args.overlap,
                )
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(images)
            n_rows += len(images)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_rows, 1),
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - started,
        }
        if args.final_fit:
            history.append(row)
            print(json.dumps(row))
            continue

        val_f1, _, _, _ = evaluate(
            model,
            eval_loader,
            device,
            args.mode,
            args.tile_width,
            args.overlap,
            args.tta,
        )
        row["inner_macro_f1_at_0.5"] = val_f1
        history.append(row)
        print(json.dumps(row))

        if val_f1 > best_f1:
            best_f1, best_epoch, epochs_without_gain = val_f1, epoch, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": args.arch,
                    "mode": args.mode,
                    "tile_width": args.tile_width,
                    "overlap": args.overlap,
                    "tta": args.tta,
                    "epoch": epoch,
                    "inner_macro_f1_at_0.5": val_f1,
                    "seed": SEED,
                },
                checkpoint,
            )
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= args.patience:
                print(f"early stop after {args.patience} epochs without improvement")
                break

    if args.final_fit:
        best_epoch = args.epochs
        torch.save(
            {
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "mode": args.mode,
                "tile_width": args.tile_width,
                "overlap": args.overlap,
                "tta": args.tta,
                "epoch": args.epochs,
                "seed": SEED,
                "final_fit": True,
                "thresholds_file": args.thresholds,
            },
            checkpoint,
        )

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["state_dict"])
    val_f1, probs, labels, image_ids = evaluate(
        model,
        eval_loader,
        device,
        args.mode,
        args.tile_width,
        args.overlap,
        args.tta,
    )
    artifact_prefix = "outer" if args.final_fit else "inner"
    np.save(output / "metrics" / f"{artifact_prefix}_probs_{args.tag}.npy", probs)
    np.save(output / "metrics" / f"{artifact_prefix}_labels_{args.tag}.npy", labels)
    pd.DataFrame({"image_id": image_ids}).to_csv(
        output / "metrics" / f"{artifact_prefix}_ids_{args.tag}.csv", index=False
    )
    report_metrics = None
    if args.final_fit:
        threshold_payload = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
        if threshold_payload.get("source_split") != "inner_calibration":
            raise ValueError("threshold file was not produced from inner_calibration")
        report_metrics = evaluate_multilabel(labels, probs, threshold_payload["thresholds"])
    result = {
        "tag": args.tag,
        "arch": args.arch,
        "mode": args.mode,
        "tile_width": args.tile_width if args.mode == "tiles" else None,
        "overlap": args.overlap if args.mode == "tiles" else None,
        "horizontal_flip_tta": args.tta,
        "seed": SEED,
        "train_split": args.train_split,
        "selection_split": args.eval_split,
        "best_epoch": best_epoch,
        "best_inner_macro_f1_at_0.5": None if args.final_fit else val_f1,
        "final_outer_metrics": report_metrics,
        "history": history,
        "checkpoint": str(checkpoint),
    }
    (output / "metrics" / f"history_{args.tag}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index.csv")
    parser.add_argument("--images-dir", default="data/train_images")
    parser.add_argument("--train-split", default="splits/inner_train.csv")
    parser.add_argument("--eval-split", default="splits/inner_calibration.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--arch", choices=["effnetb2", "resnet50"], default="effnetb2")
    parser.add_argument(
        "--mode", choices=["resize_800", "full_width", "tiles"], default="full_width"
    )
    parser.add_argument("--tile-width", type=int, default=896)
    parser.add_argument("--overlap", type=int, default=192)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--allow-outer-val", action="store_true")
    parser.add_argument(
        "--final-fit",
        action="store_true",
        help="fit for exactly --epochs on full outer train, then score outer val once",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="inner-calibrated threshold JSON; required by --final-fit",
    )
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
