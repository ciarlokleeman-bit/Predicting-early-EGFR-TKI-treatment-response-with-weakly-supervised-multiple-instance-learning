"""Fold-mean and pooled summaries of a cross-validation run."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .bootstrap import bootstrap_auc_ci
from .calibration import calibration_summary
from .metrics import confusion_counts, fold_mean_sd, metrics_from_counts, roc_auc, threshold_at_specificity

FOLD_METRICS: List[str] = ["auc", "balanced_accuracy", "f1", "sensitivity", "specificity", "ppv", "npv", "accuracy"]


def pooled_metrics(oof: pd.DataFrame) -> Dict[str, float]:
    """Metrics on concatenated out-of-fold predictions using each fold's own operating point."""
    y = oof["y_true"].to_numpy().astype(int)
    s = oof["score"].to_numpy().astype(float)
    if "pred" in oof:
        pred = oof["pred"].to_numpy().astype(int)
    else:
        pred = (s >= oof["threshold"].to_numpy()).astype(int)
    counts = confusion_counts(y, pred)
    out: Dict[str, float] = {"auc": roc_auc(y, s)}
    out.update(metrics_from_counts(**counts))
    out.update(counts)
    out.update({"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())})
    return out


def high_specificity_point(oof: pd.DataFrame, specificity: float = 0.95) -> Dict[str, float]:
    y = oof["y_true"].to_numpy().astype(int)
    s = oof["score"].to_numpy().astype(float)
    thr = threshold_at_specificity(y, s, specificity)
    flagged = s >= thr
    return {
        "target_specificity": float(specificity),
        "threshold": float(thr),
        "n_flagged": int(flagged.sum()),
        "true_positives": int((flagged & (y == 1)).sum()),
        "sensitivity": float((flagged & (y == 1)).sum() / max(y.sum(), 1)),
        "specificity": float((~flagged & (y == 0)).sum() / max((y == 0).sum(), 1)),
    }


def summarise_cv(
    fold_metrics: pd.DataFrame,
    oof: pd.DataFrame,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    ece_bins: int = 10,
    high_specificity: float = 0.95,
    calibration_fit: str = "linear",
    seed: int = 42,
) -> Dict[str, object]:
    fold_mean: Dict[str, float] = {}
    fold_sd: Dict[str, float] = {}
    for m in FOLD_METRICS:
        if m in fold_metrics:
            mean, sd = fold_mean_sd(fold_metrics[m].to_numpy())
            fold_mean[m] = mean
            fold_sd[m] = sd

    pooled = pooled_metrics(oof)
    _, lo, hi = bootstrap_auc_ci(oof["y_true"], oof["score"], n_bootstrap, ci_level, seed)
    pooled["auc_ci"] = [lo, hi]
    pooled["auc_ci_level"] = ci_level
    pooled["bootstrap_replicates"] = int(n_bootstrap)

    per_fold = fold_metrics.sort_values("fold")[["fold"] + [m for m in FOLD_METRICS if m in fold_metrics]]
    summary: Dict[str, object] = {
        "n_folds": int(len(fold_metrics)),
        "fold_mean": fold_mean,
        "fold_sd": fold_sd,
        "per_fold": per_fold.to_dict(orient="records"),
        "pooled": pooled,
        "high_specificity_point": high_specificity_point(oof, high_specificity),
        "calibration": calibration_summary(oof["y_true"], oof["score"], ece_bins, calibration_fit),
    }
    if "best_epoch" in fold_metrics:
        summary["best_epochs"] = fold_metrics.sort_values("fold")["best_epoch"].astype(int).tolist()
    if "train_time_s" in fold_metrics:
        summary["train_time_h_total"] = float(np.nansum(fold_metrics["train_time_s"].to_numpy()) / 3600.0)
    return summary
