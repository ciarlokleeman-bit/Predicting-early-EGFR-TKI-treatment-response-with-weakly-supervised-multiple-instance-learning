"""Clinical record handling: response labels, TP53 status, stage, TKI generation.

Early poor response (label = 1) is progressive disease at 12 weeks, or stable
disease with less than a 10 % reduction in the sum of target-lesion diameters.
Early favourable response is CR, PR, or SD with at least a 10 % reduction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

LABEL_COLUMNS: Dict[str, str] = {
    "primary": "label",
    "pd_only": "label_pd_only",
    "pd_plus_sd": "label_pd_plus_sd",
}

SD_REDUCTION_THRESHOLD_PCT = 10.0     # SD with < 10 % reduction counts as poor response
P53_IHC_CUTOFF_PCT = 10.0             # nuclear staining cutoff for p53 (clone DO-7)

FIRST_GENERATION_TKI = {"gefitinib", "erlotinib", "icotinib"}
THIRD_GENERATION_TKI = {"osimertinib"}

RECIST_CATEGORIES = ("CR", "PR", "SD", "PD")


def label_column(scheme: str) -> str:
    try:
        return LABEL_COLUMNS[scheme]
    except KeyError as exc:
        raise ValueError(f"unknown label scheme {scheme!r}; choose from {sorted(LABEL_COLUMNS)}") from exc


# ------------------------------------------------------------------ labels
def primary_label(recist: str, diameter_change_pct: float) -> int:
    recist = str(recist).strip().upper()
    if recist not in RECIST_CATEGORIES:
        raise ValueError(f"unexpected RECIST category {recist!r}")
    if recist == "PD":
        return 1
    if recist == "SD":
        # shrinkage is negative; a reduction smaller than 10 % means change > -10
        return int(float(diameter_change_pct) > -SD_REDUCTION_THRESHOLD_PCT)
    return 0


def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    recist = df["recist_12w"].astype(str).str.strip().str.upper()
    change = df["diameter_change_pct"].astype(float)
    df["label"] = [primary_label(r, c) for r, c in zip(recist, change)]
    df["label_pd_only"] = (recist == "PD").astype(int)
    df["label_pd_plus_sd"] = recist.isin(["PD", "SD"]).astype(int)
    return df


# ------------------------------------------------------------- covariates
def derive_tp53(df: pd.DataFrame, ihc_cutoff_pct: float = P53_IHC_CUTOFF_PCT) -> pd.DataFrame:
    df = df.copy()
    has_ngs = df["tp53_ngs"].notna() if "tp53_ngs" in df else pd.Series(False, index=df.index)
    status = np.full(len(df), np.nan)
    source = np.full(len(df), "", dtype=object)
    if "tp53_ngs" in df:
        status[has_ngs.values] = df.loc[has_ngs, "tp53_ngs"].astype(float).values
        source[has_ngs.values] = "ngs"
    if "p53_ihc_pct" in df:
        ihc = df["p53_ihc_pct"].astype(float)
        use_ihc = (~has_ngs) & ihc.notna()
        status[use_ihc.values] = (ihc[use_ihc] >= ihc_cutoff_pct).astype(float).values
        source[use_ihc.values] = "ihc"
    if np.isnan(status).any():
        missing = df.loc[np.isnan(status), "patient_id"].tolist()
        raise ValueError(f"TP53 status could not be derived for patients {missing}")
    df["tp53_status"] = status.astype(int)
    df["tp53_source"] = source
    return df


def derive_stage_binary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    stage = df["stage"].astype(str).str.strip().str.upper().str.replace("STAGE", "", regex=False).str.strip()
    stage = stage.str.replace(r"[ABC]$", "", regex=True)
    mapping = {"I": 0, "II": 0, "III": 1, "IV": 1}
    unknown = sorted(set(stage) - set(mapping))
    if unknown:
        raise ValueError(f"unrecognised stage values {unknown}")
    df["stage_binary"] = stage.map(mapping).astype(int)
    return df


def derive_tki_generation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tki = df["tki"].astype(str).str.strip().str.lower()
    gen = np.where(tki.isin(FIRST_GENERATION_TKI), "first", np.where(tki.isin(THIRD_GENERATION_TKI), "third", ""))
    if (gen == "").any():
        unknown = sorted(set(tki[gen == ""]))
        raise ValueError(f"unrecognised TKI agents {unknown}")
    df["tki_generation"] = gen
    return df


def count_tiles(df: pd.DataFrame, tiles_root: Union[str, Path]) -> pd.DataFrame:
    df = df.copy()
    counts = []
    for slide_id in df["slide_id"].astype(str):
        coords = Path(tiles_root) / slide_id / "coords.csv"
        counts.append(int(len(pd.read_csv(coords))) if coords.exists() else np.nan)
    df["n_patches"] = counts
    return df


def prepare_manifest(df: pd.DataFrame, tiles_root: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Add every derived column defined in data/manifests/schema.yaml."""
    required = ["patient_id", "slide_id", "recist_12w", "diameter_change_pct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"manifest is missing required columns {missing}")
    if df["patient_id"].duplicated().any():
        dup = df.loc[df["patient_id"].duplicated(), "patient_id"].tolist()
        raise ValueError(f"duplicate patient ids {dup}")
    df = derive_labels(df)
    if "tp53_ngs" in df or "p53_ihc_pct" in df:
        df = derive_tp53(df)
    if "stage" in df:
        df = derive_stage_binary(df)
    if "tki" in df:
        df = derive_tki_generation(df)
    if tiles_root is not None:
        df = count_tiles(df, tiles_root)
    return df


def load_manifest(path: Union[str, Path], require_derived: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["patient_id"] = df["patient_id"].astype(str)
    df["slide_id"] = df["slide_id"].astype(str)
    if require_derived and "label" not in df.columns:
        df = prepare_manifest(df)
    return df


# --------------------------------------------------------------- weighting
def class_weights(labels: Sequence[int]) -> Tuple[float, float]:
    """Return ``(w_R, w_S)`` with ``w_R = N / (2 N_R)`` and ``w_S = N / (2 N_S)``.

    ``R`` is the early poor-response class (label 1) and ``S`` the favourable class.
    """
    y = np.asarray(labels).astype(int)
    n_total = float(len(y))
    n_pos = float(y.sum())
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("both classes must be present to compute class weights")
    return n_total / (2.0 * n_pos), n_total / (2.0 * n_neg)


def group_summary(df: pd.DataFrame, label_col: str = "label") -> Dict[str, int]:
    y = df[label_col].astype(int)
    return {"n": int(len(y)), "poor": int(y.sum()), "favorable": int((1 - y).sum())}
