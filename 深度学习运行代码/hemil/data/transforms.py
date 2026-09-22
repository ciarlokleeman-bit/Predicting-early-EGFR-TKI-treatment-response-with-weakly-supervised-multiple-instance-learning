"""Tile-level augmentation used while training and when building augmented feature views.

Random horizontal / vertical flips, rotation by a multiple of 90 degrees, colour
jitter (brightness +-0.1, contrast +-0.1, saturation +-0.05) and random erasing
with probability 0.1. Tiles leave the pipeline as uint8 tensors; ImageNet
normalisation happens on the GPU inside the encoder wrapper.
"""

from __future__ import annotations

import random
from typing import Callable, Optional, Sequence

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF


class RandomRightAngleRotation:
    """Rotate by 0, 90, 180 or 270 degrees with equal probability."""

    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        if k == 0:
            return img
        return img.rotate(90 * k, expand=False)


class RandomFlips:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img = TF.hflip(img)
        if random.random() < self.p:
            img = TF.vflip(img)
        return img


def build_tile_transform(
    train: bool,
    input_size: int = 512,
    augmentation: bool = True,
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.05,
    erasing_p: float = 0.1,
) -> Callable[[Image.Image], torch.Tensor]:
    ops: list = []
    if input_size != 512:
        ops.append(T.Resize(input_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True))
    if train and augmentation:
        ops.extend(
            [
                RandomFlips(0.5),
                RandomRightAngleRotation(),
                T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation),
            ]
        )
    ops.append(T.PILToTensor())  # uint8 [3, H, W]
    if train and augmentation and erasing_p > 0:
        ops.append(T.RandomErasing(p=erasing_p, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0))
    return T.Compose(ops)


class Normalize:
    """ImageNet-style normalisation of a uint8 batch on the current device."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        return (x - mean) / std


def to_uint8_tensor(img: Image.Image, input_size: Optional[int] = None) -> torch.Tensor:
    if input_size is not None and img.size != (input_size, input_size):
        img = img.resize((input_size, input_size), Image.BILINEAR)
    return TF.pil_to_tensor(img)
