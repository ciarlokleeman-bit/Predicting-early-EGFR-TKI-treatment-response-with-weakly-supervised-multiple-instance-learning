"""Progression-free survival by the 12-week early-response label."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import median_survival_times


def pfs_by_group(
    df: pd.DataFrame,
    time_col: str = "pfs_months",
    event_col: str = "pfs_event",
    group_col: str = "label",
    reference_group: int = 1,
    ci_level: float = 0.95,
) -> Dict[str, object]:
    d = df[[time_col, event_col, group_col]].dropna().copy()
    d[event_col] = d[event_col].astype(int)
    groups = sorted(d[group_col].unique().tolist())
    if len(groups) != 2:
        raise ValueError(f"expected two groups in {group_col!r}, found {groups}")

    per_group: Dict[str, object] = {}
    curves: Dict[str, pd.DataFrame] = {}
    for g in groups:
        sub = d[d[group_col] == g]
        kmf = KaplanMeierFitter(alpha=1 - ci_level)
        kmf.fit(sub[time_col], sub[event_col], label=str(g))
        med_ci = median_survival_times(kmf.confidence_interval_)
        per_group[str(g)] = {
            "n": int(len(sub)),
            "events": int(sub[event_col].sum()),
            "median_pfs": float(kmf.median_survival_time_),
            "median_ci": [float(med_ci.iloc[0, 0]), float(med_ci.iloc[0, 1])],
        }
        curves[str(g)] = kmf.survival_function_.join(kmf.confidence_interval_)

    a = d[d[group_col] == groups[0]]
    b = d[d[group_col] == groups[1]]
    lr = logrank_test(a[time_col], b[time_col], a[event_col], b[event_col])

    cox_df = d.copy()
    # hazard ratio for the non-reference group relative to the reference (poor-response) group
    cox_df["group_vs_reference"] = (cox_df[group_col] != reference_group).astype(int)
    cph = CoxPHFitter(alpha=1 - ci_level)
    cph.fit(cox_df[[time_col, event_col, "group_vs_reference"]], duration_col=time_col, event_col=event_col)
    hr = float(np.exp(cph.params_["group_vs_reference"]))
    ci = cph.confidence_intervals_.loc["group_vs_reference"].to_numpy()
    return {
        "groups": per_group,
        "logrank_p": float(lr.p_value),
        "logrank_statistic": float(lr.test_statistic),
        "hazard_ratio": {
            "comparison": f"{group_col} != {reference_group} versus {group_col} == {reference_group}",
            "hr": hr,
            "ci": [float(np.exp(ci[0])), float(np.exp(ci[1]))],
            "p_value": float(cph.summary.loc["group_vs_reference", "p"]),
        },
        "curves": curves,
    }
