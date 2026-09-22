"""Thin OpenSlide wrapper that reads regions at a requested microns-per-pixel."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

try:  # openslide is optional at import time so that feature-cache runs do not need it
    import openslide
except ImportError:  # pragma: no cover
    openslide = None

_DEFAULT_MPP = 0.25  # Aperio AT2 at 40x


class SlideReader:
    def __init__(self, path: Union[str, Path], default_mpp: float = _DEFAULT_MPP):
        if openslide is None:
            raise ImportError("openslide-python is required to read whole-slide images")
        self.path = Path(path)
        self.slide = openslide.OpenSlide(str(self.path))
        props = self.slide.properties
        mpp_x = props.get(openslide.PROPERTY_NAME_MPP_X)
        mpp_y = props.get(openslide.PROPERTY_NAME_MPP_Y)
        if mpp_x is None or mpp_y is None:
            self.mpp = float(default_mpp)
        else:
            self.mpp = float((float(mpp_x) + float(mpp_y)) / 2.0)
        self.level_dimensions = self.slide.level_dimensions
        self.level_downsamples = [float(d) for d in self.slide.level_downsamples]
        self.width, self.height = self.level_dimensions[0]

    # ------------------------------------------------------------------ levels
    def downsample_for_mpp(self, target_mpp: float) -> float:
        return float(target_mpp) / self.mpp

    def level_for_mpp(self, target_mpp: float) -> Tuple[int, float]:
        """Highest-resolution pyramid level whose downsample does not exceed the target.

        Returns ``(level, residual_scale)`` where ``residual_scale`` is the extra
        shrink factor needed after reading from that level.
        """
        wanted = self.downsample_for_mpp(target_mpp)
        level = 0
        for i, ds in enumerate(self.level_downsamples):
            if ds <= wanted * 1.01:
                level = i
        residual = wanted / self.level_downsamples[level]
        return level, residual

    # ----------------------------------------------------------------- reading
    def read_region_at_mpp(self, x0: int, y0: int, size: int, target_mpp: float) -> Image.Image:
        """Read a ``size x size`` field at ``target_mpp`` whose top-left is ``(x0, y0)`` in level-0 pixels."""
        level, residual = self.level_for_mpp(target_mpp)
        read_size = int(math.ceil(size * residual))
        region = self.slide.read_region((int(x0), int(y0)), level, (read_size, read_size)).convert("RGB")
        if read_size != size:
            region = region.resize((size, size), Image.BILINEAR)
        return region

    def thumbnail(self, target_mpp: float) -> Image.Image:
        ds = self.downsample_for_mpp(target_mpp)
        size = (max(1, int(self.width / ds)), max(1, int(self.height / ds)))
        return self.slide.get_thumbnail(size).convert("RGB")

    def thumbnail_array(self, target_mpp: float) -> np.ndarray:
        return np.asarray(self.thumbnail(target_mpp))

    def close(self) -> None:
        self.slide.close()

    def __enter__(self) -> "SlideReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"SlideReader({self.path.name}, {self.width}x{self.height}, mpp={self.mpp:.3f})"


def open_slide(path: Union[str, Path], default_mpp: Optional[float] = None) -> SlideReader:
    return SlideReader(path, default_mpp or _DEFAULT_MPP)
