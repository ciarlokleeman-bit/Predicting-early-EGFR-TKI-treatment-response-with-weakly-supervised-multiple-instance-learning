"""Tissue segmentation from a low-resolution thumbnail.

Otsu's threshold is applied to the saturation channel of the HSV image;
saturated (stained) pixels are tissue, pale glass is background.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import color, filters, morphology


def tissue_mask_from_rgb(
    rgb: np.ndarray,
    min_object_area_px: int = 64,
    closing_radius: int = 2,
    fill_holes: bool = True,
) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an RGB thumbnail, got shape {rgb.shape}")
    hsv = color.rgb2hsv(rgb)
    saturation = hsv[..., 1]
    threshold = filters.threshold_otsu(saturation)
    mask = saturation > threshold
    if closing_radius > 0:
        mask = morphology.binary_closing(mask, morphology.disk(closing_radius))
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)
    if min_object_area_px > 0:
        mask = morphology.remove_small_objects(mask, min_size=int(min_object_area_px))
    return mask.astype(bool)


def tissue_fraction_map(mask: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Fraction of tissue pixels inside every ``window x window`` box on a ``stride`` grid.

    Uses an integral image so the cost does not depend on the window size.
    """
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    h, w = mask.shape
    integral = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys = np.arange(0, max(h - window, 0) + 1, stride)
    xs = np.arange(0, max(w - window, 0) + 1, stride)
    out = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for i, y in enumerate(ys):
        y1 = y + window
        for j, x in enumerate(xs):
            x1 = x + window
            total = integral[y1, x1] - integral[y, x1] - integral[y1, x] + integral[y, x]
            out[i, j] = total / float(window * window)
    return out


def mask_fraction(mask: np.ndarray, y0: int, x0: int, h: int, w: int) -> float:
    y1 = min(mask.shape[0], y0 + h)
    x1 = min(mask.shape[1], x0 + w)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    crop = mask[y0:y1, x0:x1]
    return float(crop.mean())
