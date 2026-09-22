"""Pairwise comparison of cross-validation runs.

Inference uses the five fold-wise AUC differences (exact two-sided Wilcoxon
signed-rank test) and Holm's step-down correction over the planned
comparisons. Because five paired differences cannot give an exact Wilcoxon p
below 0.0625, the mean fold-wise delta AUC is reported alongside. DeLong tests
on the pooled out-of-fold scores are computed as a sensitivity column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .delong import delong_roc_test
from .metrics import fold_mean_sd


def holm_correction(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    ok = ~np.isnan(p)
    adjusted = np.full_like(p, np.nan)
    if ok.any():
        adjusted[ok] = multipletests(p[ok], method="holm")[1]
    return adjusted


def wilcoxon_exact(differences: Sequence[float]) -> float:
    d = np.asarray(differences, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return float("nan")
    try:
        return float(stats.wilcoxon(d, alternative="two-sided", method="exact", zero_method="wilcox").pvalue)
    except TypeError:  # scipy < 1.9 uses `mode`
        return float(stats.wilcoxon(d, alternative="two-sided", mode="exact", zero_method="wilcox").pvalue)


def _load_run(run_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    fold_metrics = pd.read_csv(run_dir / "fold_metrics.csv").sort_values("fold")
    oof = pd.read_csv(run_dir / "oof_predictions.csv")
    oof["patient_id"] = oof["patient_id"].astype(str)
    return {"fold_metrics": fold_metrics, "oof": oof.set_index("patient_id").sort_index()}


def compare_runs(
    reference_dir: Union[str, Path],
    comparison_dirs: Mapping[str, Union[str, Path]],
    holm_on: str = "delong",
) -> pd.DataFrame:
    ref = _load_run(reference_dir)
    ref_auc = ref["fold_metrics"]["auc"].to_numpy()
    rows: List[dict] = []
    for name, run_dir in comparison_dirs.items():
        cmp = _load_run(run_dir)
        cmp_auc = cmp["fold_metrics"]["auc"].to_numpy()
        if len(cmp_auc) != len(ref_auc):
            raise ValueError(f"{name}: {len(cmp_auc)} folds versus {len(ref_auc)} in the reference run")
        delta = cmp_auc - ref_auc
        d_mean, d_sd = fold_mean_sd(delta)
        common = ref["oof"].index.intersection(cmp["oof"].index)
        if len(common) != len(ref["oof"]):
            raise ValueError(f"{name}: out-of-fold patients do not match the reference run")
        y = ref["oof"].loc[common, "y_true"].to_numpy()
        if not np.array_equal(y, cmp["oof"].loc[common, "y_true"].to_numpy()):
            raise ValueError(f"{name}: labels differ from the reference run")
        delong = delong_roc_test(y, cmp["oof"].loc[common, "score"].to_numpy(), ref["oof"].loc[common, "score"].to_numpy())
        mean_auc, sd_auc = fold_mean_sd(cmp_auc)
        rows.append(
            {
                "model": name,
                "auc_mean": mean_auc,
                "auc_sd": sd_auc,
                "balanced_accuracy_mean": fold_mean_sd(cmp["fold_metrics"]["balanced_accuracy"])[0],
                "f1_mean": fold_mean_sd(cmp["fold_metrics"]["f1"])[0],
                "sensitivity_mean": fold_mean_sd(cmp["fold_metrics"]["sensitivity"])[0],
                "specificity_mean": fold_mean_sd(cmp["fold_metrics"]["specificity"])[0],
                "ppv_mean": fold_mean_sd(cmp["fold_metrics"]["ppv"])[0],
                "npv_mean": fold_mean_sd(cmp["fold_metrics"]["npv"])[0],
                "delta_auc_mean": d_mean,
                "delta_auc_sd": d_sd,
                "delta_auc_folds": ";".join(f"{d:+.4f}" for d in delta),
                "wilcoxon_exact_p": wilcoxon_exact(delta),
                "delong_pooled_delta_auc": delong["delta_auc"],
                "delong_p_uncorrected": delong["p_value"],
            }
        )
    table = pd.DataFrame(rows)
    if len(table):
        source = "delong_p_uncorrected" if holm_on == "delong" else "wilcoxon_exact_p"
        table["p_holm"] = holm_correction(table[source].to_numpy())
        table["holm_on"] = source
    ref_mean, ref_sd = fold_mean_sd(ref_auc)
    table.attrs["reference"] = {"run": str(reference_dir), "auc_mean": ref_mean, "auc_sd": ref_sd, "n_folds": int(len(ref_auc))}
    return table
