"""Subgroup AUCs (EGFR subtype, TKI generation) and RECIST-only relabelings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.clinical import label_column
from .bootstrap import bootstrap_auc_ci
from .metrics import confusion_counts, metrics_from_counts


def _join(oof: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    o = oof.copy()
    o["patient_id"] = o["patient_id"].astype(str)
    m = manifest.copy()
    m["patient_id"] = m["patient_id"].astype(str)
    keep = [c for c in m.columns if c not in o.columns or c == "patient_id"]
    return o.merge(m[keep], on="patient_id", how="inner", validate="one_to_one")


def _evaluate(df: pd.DataFrame, y_col: str, name: str, n_bootstrap: int, ci_level: float, seed: int) -> Dict[str, object]:
    y = df[y_col].to_numpy().astype(int)
    s = df["score"].to_numpy().astype(float)
    pred = df["pred"].to_numpy().astype(int) if "pred" in df else (s >= df["threshold"].to_numpy()).astype(int)
    auc, lo, hi = bootstrap_auc_ci(y, s, n_bootstrap, ci_level, seed) if len(np.unique(y)) > 1 else (np.nan, np.nan, np.nan)
    at_point = metrics_from_counts(**confusion_counts(y, pred))
    return {
        "name": name,
        "n": int(len(y)),
        "n_poor": int(y.sum()),
        "n_favorable": int(len(y) - y.sum()),
        "auc": auc,
        "auc_ci_lower": lo,
        "auc_ci_upper": hi,
        "sensitivity": at_point["sensitivity"],
        "specificity": at_point["specificity"],
        "ppv": at_point["ppv"],
        "npv": at_point["npv"],
    }


def subgroup_table(
    oof: pd.DataFrame,
    manifest: pd.DataFrame,
    subgroups: Sequence[Mapping],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    df = _join(oof, manifest)
    rows: List[dict] = []
    for sg in subgroups:
        sel = df[df[str(sg["column"])].astype(str) == str(sg["value"])]
        row = _evaluate(sel, "y_true", str(sg["name"]), n_bootstrap, ci_level, seed)
        row.update({"column": str(sg["column"]), "value": str(sg["value"])})
        rows.append(row)
    rows.append({**_evaluate(df, "y_true", "primary_label_all", n_bootstrap, ci_level, seed), "column": "", "value": ""})
    return pd.DataFrame(rows)


def label_sensitivity_table(
    oof: pd.DataFrame,
    manifest: pd.DataFrame,
    schemes: Sequence[Mapping],
    retrained: Optional[Mapping[str, str]] = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Score the out-of-fold predictions against alternative RECIST-based targets.

    When a run re-trained on the alternative target exists (``retrained``), its
    own out-of-fold predictions are used instead of re-scoring the primary run.
    """
    df = _join(oof, manifest)
    rows: List[dict] = []
    for spec in schemes:
        name, scheme = str(spec["name"]), str(spec["scheme"])
        col = label_column(scheme)
        frame, source = df, "primary_run_rescored"
        if retrained and scheme in retrained:
            path = Path(str(retrained[scheme])) / "oof_predictions.csv"
            if path.exists():
                frame, source = _join(pd.read_csv(path), manifest), str(path)
        frame = frame.assign(y_true=frame[col].astype(int))
        row = _evaluate(frame, "y_true", name, n_bootstrap, ci_level, seed)
        row.update({"scheme": scheme, "source": source})
        rows.append(row)
    return pd.DataFrame(rows)
