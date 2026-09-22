"""Calibration of the sigmoid poor-response score."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def brier_score(y_true: Sequence, prob: Sequence) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true: Sequence, prob: Sequence, n_bins: int = 10) -> float:
    """Equal-width binning of predicted probability; weighted mean |observed - predicted|."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(y[in_bin].mean() - p[in_bin].mean())
    return float(ece)


def reliability_table(y_true: Sequence, prob: Sequence, n_bins: int = 10) -> Dict[str, list]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: Dict[str, list] = {"bin_lower": [], "bin_upper": [], "n": [], "mean_predicted": [], "observed_fraction": []}
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        rows["bin_lower"].append(float(lo))
        rows["bin_upper"].append(float(hi))
        rows["n"].append(int(in_bin.sum()))
        rows["mean_predicted"].append(float(p[in_bin].mean()) if in_bin.any() else float("nan"))
        rows["observed_fraction"].append(float(y[in_bin].mean()) if in_bin.any() else float("nan"))
    return rows


def calibration_slope_intercept(y_true: Sequence, prob: Sequence, method: str = "linear") -> Tuple[float, float]:
    """Recalibration of predicted probability on observed labels.

    ``linear``   ordinary least squares of y on p (slope 1 / intercept 0 is perfect).
    ``logistic`` logistic regression of y on logit(p) (Cox calibration slope).
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    if method == "linear":
        slope, intercept = np.polyfit(p, y, deg=1)
        return float(slope), float(intercept)
    if method == "logistic":
        import statsmodels.api as sm

        x = sm.add_constant(np.log(p / (1 - p)))
        fit = sm.Logit(y, x).fit(disp=0, maxiter=200)
        return float(fit.params[1]), float(fit.params[0])
    raise ValueError("method must be 'linear' or 'logistic'")


def calibration_summary(y_true: Sequence, prob: Sequence, n_bins: int = 10, method: str = "linear") -> Dict[str, object]:
    slope, intercept = calibration_slope_intercept(y_true, prob, method)
    return {
        "brier": brier_score(y_true, prob),
        "ece": expected_calibration_error(y_true, prob, n_bins),
        "ece_bins": int(n_bins),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "calibration_fit": method,
        "reliability": reliability_table(y_true, prob, n_bins),
    }
