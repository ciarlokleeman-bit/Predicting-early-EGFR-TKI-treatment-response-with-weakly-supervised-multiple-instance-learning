"""On-disk format for frozen-encoder feature caches.

One ``.pt`` file per slide holding a dict:

    features  fp16 tensor [views, n_tiles, dim]   view 0 is the un-augmented tile
    coords    int32 tensor [n_tiles, 2]           level-0 (x, y) of each tile
    tile_ids  list[str]
    encoder   str
    mpp       float
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import torch


def feature_path(features_root: Union[str, Path], encoder_name: str, slide_id: str) -> Path:
    return Path(features_root) / encoder_name / f"{slide_id}.pt"


def save_feature_cache(
    path: Union[str, Path],
    features: torch.Tensor,
    coords: Union[np.ndarray, torch.Tensor],
    tile_ids: Sequence[str],
    encoder_name: str,
    mpp: float = 0.5,
) -> None:
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if features.ndim != 3:
        raise ValueError(f"features must be [views, n_tiles, dim], got {tuple(features.shape)}")
    coords_t = torch.as_tensor(np.asarray(coords), dtype=torch.int32)
    if coords_t.shape[0] != features.shape[1]:
        raise ValueError("coords and features disagree on the number of tiles")
    payload: Dict[str, object] = {
        "features": features.to(torch.float16).cpu().contiguous(),
        "coords": coords_t.cpu(),
        "tile_ids": list(tile_ids),
        "encoder": encoder_name,
        "mpp": float(mpp),
        "views": int(features.shape[0]),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_feature_cache(path: Union[str, Path]) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    feats = payload["features"]
    if feats.ndim == 2:
        payload["features"] = feats.unsqueeze(0)
    return payload


def cache_summary(features_root: Union[str, Path], encoder_name: str, slide_ids: Sequence[str]) -> Dict[str, object]:
    root = Path(features_root) / encoder_name
    present: List[str] = []
    missing: List[str] = []
    n_tiles: List[int] = []
    for sid in slide_ids:
        p = root / f"{sid}.pt"
        if p.exists():
            present.append(sid)
            n_tiles.append(int(load_feature_cache(p)["features"].shape[1]))
        else:
            missing.append(sid)
    return {
        "encoder": encoder_name,
        "present": len(present),
        "missing": missing,
        "tiles_mean": float(np.mean(n_tiles)) if n_tiles else 0.0,
        "tiles_sd": float(np.std(n_tiles, ddof=1)) if len(n_tiles) > 1 else 0.0,
    }
