"""DeLong test for two correlated ROC curves (DeLong et al. 1988; Sun & Xu 2014)."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(z)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """``predictions_sorted_transposed``: [k, n] scores with the ``m`` positives first."""
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(positive[r])
        ty[r] = _midrank(negative[r])
        tz[r] = _midrank(predictions_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_roc_variance(y_true: Sequence, y_score: Sequence) -> Tuple[float, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    order = np.argsort(-y, kind="stable")
    m = int(y.sum())
    aucs, cov = _fast_delong(s[order][None, :], m)
    return float(aucs[0]), float(cov[0, 0])


def delong_roc_test(y_true: Sequence, score_a: Sequence, score_b: Sequence) -> Dict[str, float]:
    """Two-sided DeLong test of AUC(a) == AUC(b) on the same patients."""
    y = np.asarray(y_true).astype(int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    order = np.argsort(-y, kind="stable")
    m = int(y.sum())
    preds = np.vstack([a[order], b[order]])
    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0 if diff == 0 else np.inf * np.sign(diff)
    else:
        z = diff / np.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else 0.0
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta_auc": float(diff), "z": float(z), "p_value": p}
