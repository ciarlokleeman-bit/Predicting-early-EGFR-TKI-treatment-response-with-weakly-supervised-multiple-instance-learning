"""Encode every tile of a slide with a frozen encoder and write the feature cache.

View 0 holds the un-augmented tile. Additional views apply the training
augmentation (flips, right-angle rotations, colour jitter, random erasing) so
that the attention head can be trained on augmented inputs without running the
encoder inside the training loop.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..data.features_io import save_feature_cache
from ..data.transforms import build_tile_transform
from ..utils.logging import get_logger
from .factory import FrozenEncoder

log = get_logger(__name__)


class _TileFiles(Dataset):
    def __init__(self, tile_dir: Path, coords: pd.DataFrame, transform: Callable):
        self.tile_dir = tile_dir
        self.files = coords["file"].astype(str).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = Image.open(self.tile_dir / self.files[i]).convert("RGB")
        return self.transform(img)


def _encode_files(
    encoder: FrozenEncoder,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    chunks: List[torch.Tensor] = []
    for batch in loader:
        chunks.append(encoder.encode(batch.to(device, non_blocking=True), chunk=batch_size, amp=amp).cpu())
    if not chunks:
        return torch.zeros(0, encoder.feature_dim)
    return torch.cat(chunks, dim=0)


def extract_slide_features(
    encoder: FrozenEncoder,
    tiles_root: Union[str, Path],
    slide_id: str,
    out_path: Union[str, Path],
    device: torch.device,
    views: int = 1,
    batch_size: int = 128,
    num_workers: int = 8,
    amp: bool = True,
    overwrite: bool = False,
    seed: Optional[int] = None,
) -> Optional[Path]:
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        return out_path
    tile_dir = Path(tiles_root) / slide_id
    coords_path = tile_dir / "coords.csv"
    if not coords_path.exists():
        raise FileNotFoundError(coords_path)
    coords = pd.read_csv(coords_path)
    if len(coords) == 0:
        log.warning("%s has no tiles; skipping", slide_id)
        return None
    if seed is not None:
        torch.manual_seed(seed)

    t0 = time.time()
    eval_tf = build_tile_transform(train=False, input_size=encoder.input_size)
    train_tf = build_tile_transform(train=True, input_size=encoder.input_size, augmentation=True)
    encoder = encoder.to(device)

    stacks: List[torch.Tensor] = [
        _encode_files(encoder, _TileFiles(tile_dir, coords, eval_tf), device, batch_size, num_workers, amp)
    ]
    for _ in range(max(0, int(views) - 1)):
        stacks.append(
            _encode_files(encoder, _TileFiles(tile_dir, coords, train_tf), device, batch_size, num_workers, amp)
        )
    features = torch.stack(stacks, dim=0)                     # [V, N, D]
    save_feature_cache(
        out_path,
        features,
        coords[["x", "y"]].to_numpy(),
        coords["tile_id"].astype(str).tolist(),
        encoder_name=encoder.name,
        mpp=float(coords["mpp"].iloc[0]) if "mpp" in coords else 0.5,
    )
    log.info(
        "%s: %d tiles x %d views -> %s (%.1f s)", slide_id, features.shape[1], features.shape[0], out_path.name, time.time() - t0
    )
    return out_path
