"""Patient-level bootstrap confidence intervals."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np

from .metrics import roc_auc


def bootstrap_statistic(
    fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: Sequence,
    y_score: Sequence,
    n_replicates: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    stratified: bool = True,
) -> Tuple[float, float, float, np.ndarray]:
    """Return ``(point, lower, upper, replicates)`` using percentile intervals.

    With ``stratified`` the resampling is done within each class so every
    replicate contains both classes.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    rng = np.random.default_rng(seed)
    n = len(y)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    reps = np.empty(n_replicates, dtype=float)
    for r in range(n_replicates):
        if stratified and len(pos) and len(neg):
            idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        else:
            idx = rng.choice(n, n, replace=True)
        reps[r] = fn(y[idx], s[idx])
    alpha = (1.0 - ci_level) / 2.0
    lower, upper = np.nanpercentile(reps, [100 * alpha, 100 * (1 - alpha)])
    return float(fn(y, s)), float(lower), float(upper), reps


def bootstrap_auc_ci(
    y_true: Sequence,
    y_score: Sequence,
    n_replicates: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    point, lo, hi, _ = bootstrap_statistic(roc_auc, y_true, y_score, n_replicates, ci_level, seed)
    return point, lo, hi


def bootstrap_paired_difference(
    fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: Sequence,
    score_a: Sequence,
    score_b: Sequence,
    n_replicates: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """CI for ``fn(a) - fn(b)`` with both scores resampled on the same patients."""
    y = np.asarray(y_true).astype(int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    reps = np.empty(n_replicates)
    for r in range(n_replicates):
        idx = rng.choice(n, n, replace=True)
        reps[r] = fn(y[idx], a[idx]) - fn(y[idx], b[idx])
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.nanpercentile(reps, [100 * alpha, 100 * (1 - alpha)])
    return float(fn(y, a) - fn(y, b)), float(lo), float(hi)
