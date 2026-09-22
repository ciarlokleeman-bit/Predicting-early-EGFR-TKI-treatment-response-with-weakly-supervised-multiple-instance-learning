"""Macenko stain normalisation (Macenko et al., ISBI 2009).

The reference slide is the one whose median optical density is the cohort
median, so that every other slide is pulled towards a typical staining
intensity rather than towards an extreme.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from ..utils.logging import get_logger

log = get_logger(__name__)


def rgb_to_od(rgb: np.ndarray, Io: float = 240.0) -> np.ndarray:
    rgb = rgb.reshape(-1, 3).astype(np.float64)
    return -np.log((rgb + 1.0) / Io)


def od_to_rgb(od: np.ndarray, Io: float = 240.0) -> np.ndarray:
    rgb = Io * np.exp(-od)
    return np.clip(rgb, 0, 255).astype(np.uint8)


class MacenkoNormalizer:
    def __init__(
        self,
        Io: float = 240.0,
        alpha: float = 1.0,
        beta: float = 0.15,
        luminosity_threshold: float = 0.8,
        max_concentration_percentile: float = 99.0,
    ):
        self.Io = float(Io)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.luminosity_threshold = float(luminosity_threshold)
        self.max_c_percentile = float(max_concentration_percentile)
        self.HE_ref: Optional[np.ndarray] = None       # (3, 2) reference stain matrix
        self.maxC_ref: Optional[np.ndarray] = None     # (2,) reference 99th-percentile concentrations
        self.reference_slide: Optional[str] = None

    # ----------------------------------------------------------- estimation
    def _tissue_pixels(self, rgb: np.ndarray) -> np.ndarray:
        lum = rgb.reshape(-1, 3).astype(np.float64).mean(axis=1) / 255.0
        return lum < self.luminosity_threshold

    def stain_matrix(self, rgb: np.ndarray) -> np.ndarray:
        od = rgb_to_od(rgb, self.Io)
        keep = self._tissue_pixels(rgb) & ~np.any(od < self.beta, axis=1)
        od_hat = od[keep]
        if od_hat.shape[0] < 100:
            raise ValueError("too few stained pixels to estimate stain vectors")
        _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
        basis = eigvecs[:, 1:3]
        proj = od_hat @ basis
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100.0 - self.alpha)
        v_min = basis @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v_max = basis @ np.array([np.cos(max_phi), np.sin(max_phi)])
        # hematoxylin has the larger red optical density
        if v_min[0] > v_max[0]:
            he = np.stack([v_min, v_max], axis=1)
        else:
            he = np.stack([v_max, v_min], axis=1)
        return he

    @staticmethod
    def concentrations(od: np.ndarray, he: np.ndarray) -> np.ndarray:
        c, *_ = np.linalg.lstsq(he, od.T, rcond=None)
        return c  # (2, N)

    def fit(self, rgb: np.ndarray, slide_id: Optional[str] = None) -> "MacenkoNormalizer":
        he = self.stain_matrix(rgb)
        od = rgb_to_od(rgb, self.Io)
        c = self.concentrations(od, he)
        self.HE_ref = he
        self.maxC_ref = np.array(
            [np.percentile(c[0], self.max_c_percentile), np.percentile(c[1], self.max_c_percentile)]
        )
        self.reference_slide = slide_id
        return self

    def fit_tiles(self, tiles: Iterable[np.ndarray], slide_id: Optional[str] = None) -> "MacenkoNormalizer":
        pixels = np.concatenate([np.asarray(t).reshape(-1, 3) for t in tiles], axis=0)
        return self.fit(pixels.reshape(-1, 1, 3), slide_id=slide_id)

    # ---------------------------------------------------------------- apply
    def transform(self, rgb: np.ndarray) -> np.ndarray:
        if self.HE_ref is None or self.maxC_ref is None:
            raise RuntimeError("normalizer has not been fitted")
        h, w, _ = rgb.shape
        od = rgb_to_od(rgb, self.Io)
        try:
            he = self.stain_matrix(rgb)
        except ValueError:
            return rgb.copy()  # almost empty tile; nothing to normalise
        c = self.concentrations(od, he)
        max_c = np.array(
            [np.percentile(c[0], self.max_c_percentile), np.percentile(c[1], self.max_c_percentile)]
        )
        max_c = np.where(max_c < 1e-6, 1e-6, max_c)
        c = c * (self.maxC_ref / max_c)[:, None]
        od_norm = (self.HE_ref @ c).T
        return od_to_rgb(od_norm, self.Io).reshape(h, w, 3)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return self.transform(rgb)

    # ---------------------------------------------------------- persistence
    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            HE_ref=self.HE_ref,
            maxC_ref=self.maxC_ref,
            Io=self.Io,
            alpha=self.alpha,
            beta=self.beta,
            luminosity_threshold=self.luminosity_threshold,
            max_c_percentile=self.max_c_percentile,
            reference_slide=str(self.reference_slide),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MacenkoNormalizer":
        data = np.load(path, allow_pickle=False)
        obj = cls(
            Io=float(data["Io"]),
            alpha=float(data["alpha"]),
            beta=float(data["beta"]),
            luminosity_threshold=float(data["luminosity_threshold"]),
            max_concentration_percentile=float(data["max_c_percentile"]),
        )
        obj.HE_ref = data["HE_ref"]
        obj.maxC_ref = data["maxC_ref"]
        obj.reference_slide = str(data["reference_slide"])
        return obj


# ------------------------------------------------------------------ reference
def slide_median_od(tiles: Sequence[np.ndarray], Io: float = 240.0, luminosity_threshold: float = 0.8) -> float:
    """Median optical density of stained pixels pooled over a sample of tiles."""
    ods: List[np.ndarray] = []
    for tile in tiles:
        arr = np.asarray(tile)
        lum = arr.reshape(-1, 3).mean(axis=1) / 255.0
        od = rgb_to_od(arr, Io).mean(axis=1)
        ods.append(od[lum < luminosity_threshold])
    pooled = np.concatenate(ods) if ods else np.zeros(1)
    return float(np.median(pooled)) if pooled.size else 0.0


def select_reference_slide(
    per_slide_stat: Dict[str, float],
) -> Tuple[str, float]:
    """Slide whose staining statistic is closest to the cohort median."""
    if not per_slide_stat:
        raise ValueError("no slides to choose from")
    ids = list(per_slide_stat.keys())
    values = np.array([per_slide_stat[i] for i in ids], dtype=np.float64)
    target = float(np.median(values))
    idx = int(np.argmin(np.abs(values - target)))
    return ids[idx], target


def load_tile_sample(tile_dir: Union[str, Path], n: int, rng: np.random.Generator) -> List[np.ndarray]:
    files = sorted(p for p in Path(tile_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not files:
        return []
    if len(files) > n:
        files = [files[i] for i in rng.choice(len(files), size=n, replace=False)]
    return [np.asarray(Image.open(f).convert("RGB")) for f in files]
