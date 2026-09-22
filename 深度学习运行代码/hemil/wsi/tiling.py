"""Non-overlapping 512 x 512 tile grid at 0.5 um/px restricted to tissue."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

from ..utils.logging import get_logger
from .reader import SlideReader
from .tissue import mask_fraction, tissue_mask_from_rgb

log = get_logger(__name__)


@dataclass
class TileSpec:
    x: int                 # level-0 pixel coordinates of the top-left corner
    y: int
    tissue_fraction: float

    @property
    def tile_id(self) -> str:
        return f"{self.x}_{self.y}"


class TileGrid:
    """Enumerates tissue tiles of a slide on a fixed grid at the target resolution."""

    def __init__(
        self,
        reader: SlideReader,
        target_mpp: float = 0.5,
        patch_size: int = 512,
        stride: Optional[int] = None,
        min_tissue_fraction: float = 0.70,
        thumbnail_mpp: float = 8.0,
        mask_kwargs: Optional[dict] = None,
    ):
        self.reader = reader
        self.target_mpp = float(target_mpp)
        self.patch_size = int(patch_size)
        self.stride = int(stride or patch_size)
        self.min_tissue_fraction = float(min_tissue_fraction)
        self.thumbnail_mpp = float(thumbnail_mpp)
        self.thumbnail = reader.thumbnail_array(self.thumbnail_mpp)
        self.mask = tissue_mask_from_rgb(self.thumbnail, **(mask_kwargs or {}))
        # level-0 pixels per thumbnail pixel
        self._thumb_scale = self.reader.downsample_for_mpp(self.thumbnail_mpp)
        # level-0 pixels covered by one tile
        self._tile_l0 = self.patch_size * self.reader.downsample_for_mpp(self.target_mpp)
        self._stride_l0 = self.stride * self.reader.downsample_for_mpp(self.target_mpp)

    def __iter__(self) -> Iterator[TileSpec]:
        tile_thumb = max(1, int(round(self._tile_l0 / self._thumb_scale)))
        n_rows = int(math.floor((self.reader.height - self._tile_l0) / self._stride_l0)) + 1
        n_cols = int(math.floor((self.reader.width - self._tile_l0) / self._stride_l0)) + 1
        for r in range(max(n_rows, 0)):
            y = int(round(r * self._stride_l0))
            ty = int(y / self._thumb_scale)
            for c in range(max(n_cols, 0)):
                x = int(round(c * self._stride_l0))
                tx = int(x / self._thumb_scale)
                frac = mask_fraction(self.mask, ty, tx, tile_thumb, tile_thumb)
                if frac >= self.min_tissue_fraction:
                    yield TileSpec(x=x, y=y, tissue_fraction=frac)

    def tiles(self) -> List[TileSpec]:
        return list(iter(self))

    def read(self, spec: TileSpec) -> Image.Image:
        return self.reader.read_region_at_mpp(spec.x, spec.y, self.patch_size, self.target_mpp)


def extract_slide_tiles(
    slide_path: Union[str, Path],
    slide_id: str,
    out_root: Union[str, Path],
    target_mpp: float = 0.5,
    patch_size: int = 512,
    stride: Optional[int] = None,
    min_tissue_fraction: float = 0.70,
    thumbnail_mpp: float = 8.0,
    mask_kwargs: Optional[dict] = None,
    tile_format: str = "png",
    jpeg_quality: int = 95,
    thumbnails_out: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Write every tissue tile of one slide and return its coordinate table."""
    out_dir = Path(out_root) / slide_id
    coords_path = out_dir / "coords.csv"
    if coords_path.exists() and not overwrite:
        log.info("%s: tiles already extracted, skipping", slide_id)
        return pd.read_csv(coords_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    with SlideReader(slide_path) as reader:
        grid = TileGrid(
            reader,
            target_mpp=target_mpp,
            patch_size=patch_size,
            stride=stride,
            min_tissue_fraction=min_tissue_fraction,
            thumbnail_mpp=thumbnail_mpp,
            mask_kwargs=mask_kwargs,
        )
        if thumbnails_out is not None:
            tdir = Path(thumbnails_out)
            tdir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(grid.thumbnail).save(tdir / f"{slide_id}.png")
            Image.fromarray((grid.mask * 255).astype(np.uint8)).save(tdir / f"{slide_id}_mask.png")

        records = []
        ext = "jpg" if tile_format.lower() in {"jpg", "jpeg"} else "png"
        for spec in grid:
            tile = grid.read(spec)
            fname = f"{slide_id}_{spec.tile_id}.{ext}"
            if ext == "jpg":
                tile.save(out_dir / fname, quality=jpeg_quality, subsampling=0)
            else:
                tile.save(out_dir / fname, compress_level=1)
            records.append(
                {
                    "tile_id": spec.tile_id,
                    "file": fname,
                    "x": spec.x,
                    "y": spec.y,
                    "tissue_fraction": round(spec.tissue_fraction, 4),
                    "mpp": target_mpp,
                    "size": patch_size,
                }
            )
        mpp = reader.mpp
        dims = (reader.width, reader.height)

    coords = pd.DataFrame.from_records(
        records, columns=["tile_id", "file", "x", "y", "tissue_fraction", "mpp", "size"]
    )
    coords.attrs["slide_mpp"] = mpp
    coords.attrs["slide_dims"] = dims
    coords.to_csv(coords_path, index=False)
    log.info("%s: %d tiles kept (>= %.0f%% tissue)", slide_id, len(coords), 100 * min_tissue_fraction)
    return coords
