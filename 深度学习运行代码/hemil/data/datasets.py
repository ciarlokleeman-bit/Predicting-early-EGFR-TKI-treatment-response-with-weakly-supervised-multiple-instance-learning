"""Bag datasets: one item is one patient (all or a random subset of its patches).

Two inputs are supported.

* ``FeatureBagDataset`` reads cached frozen-encoder embeddings. When several
  augmented views were stored, training draws one view per sampled tile.
* ``TileBagDataset`` reads the 512 px tiles themselves so that augmentation
  and the frozen encoder run on the fly (``data.input: tiles``).

During training 100 patches are sampled at random from each patient in every
epoch; at inference every retained patch is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..utils.seed import seed_worker
from .features_io import feature_path, load_feature_cache

PatchesPerBag = Union[int, str, None]


def _resolve_n_sample(value: PatchesPerBag) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() in {"all", "none", "full"}:
            return None
        return int(value)
    return int(value)


def sample_tile_indices(n_tiles: int, n_sample: Optional[int], rng: np.random.Generator) -> np.ndarray:
    if n_sample is None or n_tiles <= 0:
        return np.arange(n_tiles)
    replace = n_tiles < n_sample
    return np.sort(rng.choice(n_tiles, size=n_sample, replace=replace))


class _BagDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        patient_ids: Sequence[str],
        label_col: str,
        train: bool,
        patches_per_bag: PatchesPerBag,
        seed: int = 0,
    ):
        manifest = manifest.copy()
        manifest["patient_id"] = manifest["patient_id"].astype(str)
        manifest = manifest.set_index("patient_id", drop=False)
        missing = [p for p in patient_ids if p not in manifest.index]
        if missing:
            raise KeyError(f"patients missing from manifest: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        self.rows = manifest.loc[list(patient_ids)]
        self.patient_ids = [str(p) for p in patient_ids]
        self.label_col = label_col
        self.train = train
        self.n_sample = _resolve_n_sample(patches_per_bag) if train else None
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def labels(self) -> np.ndarray:
        return self.rows[self.label_col].astype(int).to_numpy()

    def _row(self, index: int) -> pd.Series:
        return self.rows.iloc[index]

    def _rng_for_item(self) -> np.random.Generator:
        # worker processes are re-seeded by seed_worker; the main process uses the dataset rng
        info = torch.utils.data.get_worker_info()
        if info is None:
            return self._rng
        return np.random.default_rng(np.random.randint(0, 2**31 - 1))


class FeatureBagDataset(_BagDataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        patient_ids: Sequence[str],
        features_root: Union[str, Path],
        encoder_name: str,
        label_col: str = "label",
        train: bool = False,
        patches_per_bag: PatchesPerBag = 100,
        augmentation: bool = True,
        seed: int = 0,
    ):
        super().__init__(manifest, patient_ids, label_col, train, patches_per_bag, seed)
        self.features_root = Path(features_root)
        self.encoder_name = encoder_name
        self.augmentation = bool(augmentation)
        for sid in self.rows["slide_id"].astype(str):
            p = feature_path(self.features_root, encoder_name, sid)
            if not p.exists():
                raise FileNotFoundError(f"feature cache not found: {p}")

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self._row(index)
        slide_id = str(row["slide_id"])
        cache = load_feature_cache(feature_path(self.features_root, self.encoder_name, slide_id))
        feats: torch.Tensor = cache["features"]            # [V, N, D] fp16
        n_views, n_tiles, _ = feats.shape
        rng = self._rng_for_item()
        idx = sample_tile_indices(n_tiles, self.n_sample, rng)
        if self.train and self.augmentation and n_views > 1:
            views = torch.as_tensor(rng.integers(0, n_views, size=len(idx)))
        else:
            views = torch.zeros(len(idx), dtype=torch.long)
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        bag = feats[views, idx_t].float()                  # [n, D]
        return {
            "features": bag,
            "label": torch.tensor(float(row[self.label_col])),
            "patient_id": str(row["patient_id"]),
            "slide_id": slide_id,
            "tile_idx": idx_t,
            "coords": cache["coords"][idx_t],
        }


class TileBagDataset(_BagDataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        patient_ids: Sequence[str],
        tiles_root: Union[str, Path],
        transform: Callable[[Image.Image], torch.Tensor],
        label_col: str = "label",
        train: bool = False,
        patches_per_bag: PatchesPerBag = 100,
        seed: int = 0,
    ):
        super().__init__(manifest, patient_ids, label_col, train, patches_per_bag, seed)
        self.tiles_root = Path(tiles_root)
        self.transform = transform
        self._coords: Dict[str, pd.DataFrame] = {}
        for sid in self.rows["slide_id"].astype(str):
            coords = self.tiles_root / sid / "coords.csv"
            if not coords.exists():
                raise FileNotFoundError(f"tile table not found: {coords}")
            self._coords[sid] = pd.read_csv(coords)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self._row(index)
        slide_id = str(row["slide_id"])
        coords = self._coords[slide_id]
        rng = self._rng_for_item()
        idx = sample_tile_indices(len(coords), self.n_sample, rng)
        tiles: List[torch.Tensor] = []
        for i in idx:
            img = Image.open(self.tiles_root / slide_id / coords.iloc[int(i)]["file"]).convert("RGB")
            tiles.append(self.transform(img))
        bag = torch.stack(tiles, dim=0) if tiles else torch.zeros(0, 3, 512, 512, dtype=torch.uint8)
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        return {
            "images": bag,                                  # uint8 [n, 3, H, W]
            "label": torch.tensor(float(row[self.label_col])),
            "patient_id": str(row["patient_id"]),
            "slide_id": slide_id,
            "tile_idx": idx_t,
            "coords": torch.as_tensor(coords.iloc[idx][["x", "y"]].to_numpy(), dtype=torch.int32),
        }


def collate_bags(items: List[Dict[str, object]]) -> Dict[str, object]:
    """Pad bags to the longest one in the batch and return a boolean tile mask."""
    key = "features" if "features" in items[0] else "images"
    lengths = [int(it[key].shape[0]) for it in items]
    n_max = max(lengths)
    first = items[0][key]
    padded = first.new_zeros((len(items), n_max) + tuple(first.shape[1:]))
    mask = torch.zeros(len(items), n_max, dtype=torch.bool)
    for b, it in enumerate(items):
        n = lengths[b]
        padded[b, :n] = it[key]
        mask[b, :n] = True
    return {
        key: padded,
        "mask": mask,
        "label": torch.stack([it["label"] for it in items]),
        "patient_id": [it["patient_id"] for it in items],
        "slide_id": [it["slide_id"] for it in items],
        "tile_idx": [it["tile_idx"] for it in items],
        "coords": [it["coords"] for it in items],
        "n_tiles": torch.tensor(lengths),
    }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_bags,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
