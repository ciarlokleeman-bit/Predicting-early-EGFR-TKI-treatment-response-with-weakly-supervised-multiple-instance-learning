"""Five-fold stratified outer cross-validation with an inner early-stopping split.

Each outer test fold keeps the 40:60 class ratio (8 poor / 12 favourable
patients). Inside every 80-patient outer training partition, 16 patients form
the inner validation set used only for early stopping and for choosing the
operating point; the remaining 64 are used for parameter updates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

FoldSplit = Dict[str, object]


def make_outer_folds(labels: Sequence[int], n_folds: int = 5, seed: int = 42, shuffle: bool = True) -> List[np.ndarray]:
    y = np.asarray(labels).astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=seed if shuffle else None)
    return [test_idx for _, test_idx in skf.split(np.zeros(len(y)), y)]


def split_inner(train_idx: np.ndarray, labels: Sequence[int], val_size: int, seed: int) -> tuple:
    y = np.asarray(labels).astype(int)[train_idx]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=int(val_size), random_state=seed)
    inner_train, inner_val = next(sss.split(np.zeros(len(train_idx)), y))
    return train_idx[inner_train], train_idx[inner_val]


def build_splits(
    manifest: pd.DataFrame,
    label_col: str = "label",
    n_folds: int = 5,
    inner_val_size: int = 16,
    seed: int = 42,
    shuffle: bool = True,
) -> List[FoldSplit]:
    ids = manifest["patient_id"].astype(str).to_numpy()
    y = manifest[label_col].astype(int).to_numpy()
    all_idx = np.arange(len(ids))
    splits: List[FoldSplit] = []
    for k, test_idx in enumerate(make_outer_folds(y, n_folds=n_folds, seed=seed, shuffle=shuffle)):
        train_all = np.setdiff1d(all_idx, test_idx)
        inner_train, inner_val = split_inner(train_all, y, inner_val_size, seed=seed + k)
        splits.append(
            {
                "fold": k,
                "seed": seed,
                "train": ids[inner_train].tolist(),
                "val": ids[inner_val].tolist(),
                "test": ids[test_idx].tolist(),
                "counts": {
                    "train": _counts(y[inner_train]),
                    "val": _counts(y[inner_val]),
                    "test": _counts(y[test_idx]),
                },
            }
        )
    _assert_disjoint(splits, len(ids))
    return splits


def _counts(y: np.ndarray) -> Dict[str, int]:
    return {"n": int(len(y)), "poor": int(y.sum()), "favorable": int(len(y) - y.sum())}


def _assert_disjoint(splits: List[FoldSplit], n_patients: int) -> None:
    seen_test: set = set()
    for s in splits:
        train, val, test = set(s["train"]), set(s["val"]), set(s["test"])
        if train & val or train & test or val & test:
            raise RuntimeError(f"overlapping partitions in fold {s['fold']}")
        if seen_test & test:
            raise RuntimeError("a patient appears in more than one outer test fold")
        seen_test |= test
    if len(seen_test) != n_patients:
        raise RuntimeError("outer test folds do not cover every patient exactly once")


def save_splits(splits: List[FoldSplit], out_dir: Union[str, Path]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in splits:
        with open(out / f"fold{s['fold']}.json", "w", encoding="utf-8") as fh:
            json.dump(s, fh, indent=2)


def load_splits(split_dir: Union[str, Path]) -> List[FoldSplit]:
    files = sorted(Path(split_dir).glob("fold*.json"), key=lambda p: int(p.stem.replace("fold", "")))
    if not files:
        raise FileNotFoundError(f"no fold*.json files in {split_dir}")
    splits = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            splits.append(json.load(fh))
    return splits
