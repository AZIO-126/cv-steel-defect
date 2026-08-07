"""Champion and challenger segmentation models, both from segmentation-models-pytorch.

Champion — U-Net. Encoder-decoder with skip connections carrying the full-resolution detail
from each encoder stage straight across to the matching decoder stage. That is the property
that matters on this dataset: phase 2 measured class 2 as both rare and small-area, and thin
scratches are exactly what a decoder loses when it has to rebuild them from a downsampled
bottleneck alone.

Challenger — DeepLabV3+. Keeps the encoder at output stride 16 and recovers context with
atrous spatial pyramid pooling — parallel dilated convolutions at several rates — instead of
a full decoder ladder. It sees more context per pixel at less cost, but restores detail from
a single low-resolution feature map plus one shallow skip, so it should trade away exactly
what U-Net's skips preserve. That is the comparison worth making.

Both use the SAME encoder (ResNet-34, ImageNet-pretrained) on purpose. Swapping the encoder
as well would make the result a statement about encoders, not architectures, and the phase
README calls that out.

ResNet-34 over ResNet-50: 1600x256 is a wide frame and both models are trained at native
resolution, so activation memory, not parameter count, is the binding constraint on one
Colab GPU. ResNet-34 leaves room for a batch large enough for stable BatchNorm statistics.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
N_CLASSES = 4


def build_unet(encoder: str = ENCODER, weights: str | None = ENCODER_WEIGHTS) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,       # grayscale replicated 3x, see seg_data.py
        classes=N_CLASSES,   # 4 independent sigmoid channels, not a 5-way softmax
    )


def build_deeplabv3p(encoder: str = ENCODER, weights: str | None = ENCODER_WEIGHTS) -> nn.Module:
    return smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=N_CLASSES,
    )


MODELS = {
    "unet": build_unet,
    "deeplabv3p": build_deeplabv3p,
}


def build(name: str, pretrained: bool = True) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, expected one of {sorted(MODELS)}")
    return MODELS[name](weights=ENCODER_WEIGHTS if pretrained else None)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
