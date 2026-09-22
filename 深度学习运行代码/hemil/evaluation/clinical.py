"""Logistic clinical baselines and incremental value of the slide-level score.

Baselines: Ki-67 alone, TP53 status alone, stage (I-II versus III-IV) and the
three together. The slide score is then added to the three-variable model and
judged by the likelihood-ratio test, the change in AUC and the category-free
net reclassification improvement. A further model adds TKI generation.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .bootstrap import bootstrap_auc_ci
from .metrics import roc_auc

TERM_COLUMNS: Dict[str, str] = {
    "ki67": "ki67",
    "tp53": "tp53_status",
    "stage": "stage_binary",
    "score": "score",
    "tki_generation": "tki_third",
}


def design_matrix(df: pd.DataFrame, terms: Sequence[str], columns: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    cols = dict(TERM_COLUMNS)
    if columns:
        cols.update({k: v for k, v in columns.items() if k in cols})
    x = pd.DataFrame(index=df.index)
    for term in terms:
        col = cols.get(term, term)
        if term == "tki_generation" and "tki_third" not in df:
            x["tki_third"] = (df["tki_generation"].astype(str).str.lower() == "third").astype(float)
        else:
            x[col] = df[col].astype(float)
    return sm.add_constant(x, has_constant="add")


def fit_logistic(df: pd.DataFrame, terms: Sequence[str], y_col: str = "label", columns: Optional[Mapping[str, str]] = None):
    x = design_matrix(df, terms, columns)
    y = df[y_col].astype(float)
    return sm.Logit(y, x).fit(disp=0, maxiter=500)


def likelihood_ratio_test(full, reduced) -> Dict[str, float]:
    chi2 = 2.0 * (full.llf - reduced.llf)
    df = int(full.df_model - reduced.df_model)
    return {"chi2": float(chi2), "df": df, "p_value": float(stats.chi2.sf(chi2, df)) if df > 0 else float("nan")}


def continuous_nri(y_true: Sequence, p_new: Sequence, p_old: Sequence) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    up = np.asarray(p_new, dtype=float) > np.asarray(p_old, dtype=float)
    down = np.asarray(p_new, dtype=float) < np.asarray(p_old, dtype=float)
    ev, ne = y == 1, y == 0
    nri_events = float(up[ev].mean() - down[ev].mean()) if ev.any() else float("nan")
    nri_nonevents = float(down[ne].mean() - up[ne].mean()) if ne.any() else float("nan")
    return {"nri": nri_events + nri_nonevents, "nri_events": nri_events, "nri_nonevents": nri_nonevents}


def nri_bootstrap(
    df: pd.DataFrame,
    full_terms: Sequence[str],
    reduced_terms: Sequence[str],
    y_col: str = "label",
    n_replicates: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Refit both models on each bootstrap sample and recompute the category-free NRI."""
    rng = np.random.default_rng(seed)
    n = len(df)
    reps = []
    for _ in range(n_replicates):
        idx = rng.integers(0, n, n)
        boot = df.iloc[idx].reset_index(drop=True)
        if boot[y_col].nunique() < 2:
            continue
        try:
            full = fit_logistic(boot, full_terms, y_col)
            red = fit_logistic(boot, reduced_terms, y_col)
        except Exception:  # separation on a small resample
            continue
        reps.append(continuous_nri(boot[y_col], full.predict(), red.predict())["nri"])
    alpha = (1 - ci_level) / 2
    lo, hi = np.percentile(reps, [100 * alpha, 100 * (1 - alpha)]) if reps else (np.nan, np.nan)
    full = fit_logistic(df, full_terms, y_col)
    red = fit_logistic(df, reduced_terms, y_col)
    point = continuous_nri(df[y_col], full.predict(), red.predict())["nri"]
    return float(point), float(lo), float(hi)


def merge_scores(manifest: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    m = manifest.copy()
    m["patient_id"] = m["patient_id"].astype(str)
    o = oof[["patient_id", "score"]].copy()
    o["patient_id"] = o["patient_id"].astype(str)
    merged = m.merge(o, on="patient_id", how="inner", validate="one_to_one")
    if len(merged) != len(oof):
        raise ValueError("out-of-fold predictions and manifest do not cover the same patients")
    if "tki_generation" in merged:
        merged["tki_third"] = (merged["tki_generation"].astype(str).str.lower() == "third").astype(float)
    return merged


def run_clinical_baseline(
    manifest: pd.DataFrame,
    oof: pd.DataFrame,
    models: Sequence[Mapping],
    incremental: Mapping,
    y_col: str = "label",
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, object]:
    df = merge_scores(manifest, oof)
    fits = {}
    rows: List[dict] = []
    for spec in models:
        name, terms = str(spec["name"]), list(spec["terms"])
        fit = fit_logistic(df, terms, y_col)
        fits[name] = fit
        p = fit.predict()
        auc, lo, hi = bootstrap_auc_ci(df[y_col], p, n_bootstrap, ci_level, seed)
        rows.append(
            {
                "model": name,
                "terms": "+".join(terms),
                "auc": auc,
                "auc_ci_lower": lo,
                "auc_ci_upper": hi,
                "llf": float(fit.llf),
                "aic": float(fit.aic),
                "n": int(len(df)),
            }
        )
    table = pd.DataFrame(rows)

    reduced_name, full_name = str(incremental["reduced"]), str(incremental["full"])
    reduced, full = fits[reduced_name], fits[full_name]
    ref_auc = float(table.loc[table.model == reduced_name, "auc"].iloc[0])
    table["delta_auc_vs_" + reduced_name] = table["auc"] - ref_auc
    lrt = likelihood_ratio_test(full, reduced)
    full_terms = [t for s in models if s["name"] == full_name for t in s["terms"]]
    reduced_terms = [t for s in models if s["name"] == reduced_name for t in s["terms"]]
    nri, nri_lo, nri_hi = nri_bootstrap(df, full_terms, reduced_terms, y_col, n_bootstrap, ci_level, seed)

    result: Dict[str, object] = {
        "table": table,
        "incremental": {
            "reduced": reduced_name,
            "full": full_name,
            "delta_auc": float(roc_auc(df[y_col], full.predict()) - roc_auc(df[y_col], reduced.predict())),
            "likelihood_ratio": lrt,
            "nri_category_free": {"nri": nri, "ci_lower": nri_lo, "ci_upper": nri_hi},
        },
        "coefficients": {name: _coef_table(fit, float(incremental.get("score_unit", 0.10))) for name, fit in fits.items()},
    }
    return result


def _coef_table(fit, score_unit: float) -> List[dict]:
    ci = fit.conf_int()
    rows = []
    for term in fit.params.index:
        beta = float(fit.params[term])
        lo, hi = float(ci.loc[term, 0]), float(ci.loc[term, 1])
        row = {"term": term, "beta": beta, "se": float(fit.bse[term]), "p_value": float(fit.pvalues[term]), "or": np.exp(beta), "or_ci": [np.exp(lo), np.exp(hi)]}
        if term == "score":
            row["or_per_unit"] = {"unit": score_unit, "or": float(np.exp(beta * score_unit)), "ci": [float(np.exp(lo * score_unit)), float(np.exp(hi * score_unit))]}
        rows.append(row)
    return rows
