"""Patient-level classification metrics."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _as_arrays(y_true: Sequence, y_score: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int).reshape(-1)
    s = np.asarray(y_score).astype(float).reshape(-1)
    if y.shape != s.shape:
        raise ValueError("y_true and y_score must have the same length")
    return y, s


def roc_auc(y_true: Sequence, y_score: Sequence) -> float:
    y, s = _as_arrays(y_true, y_score)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def youden_threshold(y_true: Sequence, y_score: Sequence) -> float:
    """Score cut-off maximising sensitivity + specificity - 1 on the given predictions."""
    y, s = _as_arrays(y_true, y_score)
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    # the first roc_curve threshold is +inf / max+1; skip it so the cut-off is attainable
    valid = np.isfinite(thr)
    j = np.where(valid, j, -np.inf)
    best = int(np.argmax(j))
    return float(thr[best])


def threshold_at_specificity(y_true: Sequence, y_score: Sequence, specificity: float = 0.95) -> float:
    """Lowest cut-off whose specificity is at least ``specificity`` (maximises sensitivity under that constraint)."""
    y, s = _as_arrays(y_true, y_score)
    fpr, tpr, thr = roc_curve(y, s)
    ok = (1.0 - fpr >= specificity) & np.isfinite(thr)
    if not ok.any():
        return float(np.max(s) + 1e-6)
    idx = np.where(ok)[0]
    best = idx[np.argmax(tpr[idx])]
    return float(thr[best])


def confusion_counts(y_true: Sequence, y_pred: Sequence) -> Dict[str, int]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    return {
        "tp": int(((p == 1) & (y == 1)).sum()),
        "fp": int(((p == 1) & (y == 0)).sum()),
        "tn": int(((p == 0) & (y == 0)).sum()),
        "fn": int(((p == 0) & (y == 1)).sum()),
    }


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    sens = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    ppv = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "accuracy": _safe_div(tp + tn, tp + fp + tn + fn),
        "f1": f1,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
    }


def binary_metrics(y_true: Sequence, y_score: Sequence, threshold: float) -> Dict[str, float]:
    """AUC plus threshold-dependent metrics at the given operating point."""
    y, s = _as_arrays(y_true, y_score)
    pred = (s >= threshold).astype(int)
    counts = confusion_counts(y, pred)
    out: Dict[str, float] = {"auc": roc_auc(y, s)}
    out.update(metrics_from_counts(**counts))
    out.update(counts)
    out.update({"threshold": float(threshold), "n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())})
    return out


def fold_mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    sd = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    return float(np.mean(v)), sd
