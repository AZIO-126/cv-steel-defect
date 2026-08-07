"""Combined Dice + BCE loss.

Why not pure BCE. A defect covers a tiny fraction of a 1600x256 frame — over the phase-1
index the defect pixels are on the order of 1-2% of the pixels in a defect-bearing image,
and 0% in the ~47% of images that are clean. BCE averages over pixels, so "predict
background everywhere" already scores well under it: the gradient pulling the model toward
the defect pixels is outweighed by the mass of easy background pixels. A model trained that
way converges to an empty prediction and reports a low loss while being useless.

Dice is computed from the overlap between prediction and target and is normalised by the
size of the two regions, so an empty prediction scores 0 no matter how much background it
got right. That fixes BCE's blind spot but on its own gives a noisy gradient early in
training and is undefined for an empty ground truth.

Combining them is the standard fix and the one used here: BCE supplies a stable, well-scaled
per-pixel gradient, Dice supplies the region-level pressure that stops the collapse to
background. `smooth` keeps the Dice term finite on all-empty channels.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """1 - Dice, averaged over the (image, class) pairs that HAVE a defect.

    Computed per (image, class) rather than over the flattened batch, because pooling lets one
    large class-3 defect dominate and the rare small classes stop contributing.

    Empty-ground-truth pairs are excluded, and that exclusion is the whole point. For a pair
    with no defect the intersection is 0, so the score reduces to `smooth / (pred.sum() +
    smooth)` — maximised by driving the prediction to zero. Those pairs are the overwhelming
    majority here: four class channels with usually at most one present, plus 47% of images
    carrying no defect at all, leaves roughly 85-90% of channel-slots empty. Averaging over
    all of them makes this term a standing instruction to predict nothing, which is the exact
    failure Dice is supposed to prevent.

    That is not hypothetical. The first U-Net run on this dataset used the all-pairs form and
    collapsed: training loss fell 0.51 -> 0.13 while validation Dice went 0.58 -> 0.09 and the
    false-positive rate fell 0.66 -> 0.10, i.e. it stopped predicting anything at all.

    Empty pairs are still fully supervised — by BCE, which is the right tool for "there is
    nothing here" and already punishes false positives. So BCE governs where not to fire and
    Dice governs covering the defect where one exists.
    """
    probs = torch.sigmoid(logits)
    dims = (2, 3)  # H, W — keep batch and class separate
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    has_defect = targets.sum(dim=dims) > 0
    if not bool(has_defect.any()):
        # A batch of entirely defect-free images: Dice has nothing to say, and inventing a
        # value here would reintroduce the pull toward empty. BCE alone supervises this batch.
        return logits.sum() * 0.0  # keeps the graph connected, contributes no gradient
    return 1.0 - dice[has_defect].mean()


class DiceBCELoss(nn.Module):
    """`bce_weight * BCE + dice_weight * (1 - Dice)`; defaults weight them equally."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = soft_dice_loss(logits, targets, self.smooth)
        return self.bce_weight * bce + self.dice_weight * dice

    def components(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Both terms separately — logged during training so the report can show that the
        Dice term is the one still moving after BCE has flattened out."""
        with torch.no_grad():
            bce = F.binary_cross_entropy_with_logits(logits, targets).item()
            dice = soft_dice_loss(logits, targets, self.smooth).item()
        return {"bce": bce, "dice_loss": dice}
