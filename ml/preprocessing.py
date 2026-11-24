"""
ml/preprocessing.py
===================
Centralised data loading, cleaning, feature engineering, and MICE imputation
for the Emergency Healthcare ED-Triage ML pipeline.

Ensures training and inference use consistent feature definitions.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Optional: MICE imputer
# ---------------------------------------------------------------------------
try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    _HAS_MICE = True
except ImportError:
    _HAS_MICE = False
    warnings.warn(
        "[warn] sklearn IterativeImputer unavailable; falling back to median imputation.",
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET_COL   = "acuity_target"
CLASSES      = [1, 2, 3, 4]

# Continuous vitals that go into MICE
MICE_COLS = [
    "age", "systolic_bp", "diastolic_bp", "heart_rate",
    "respiratory_rate", "temperature", "spo2", "pain_score",
]

# Age bucket boundaries and names
AGE_BINS   = [0, 17, 39, 64, 79, 120]
AGE_LABELS = ["age_0_17", "age_18_39", "age_40_64", "age_65_79", "age_80plus"]

INPUT_BOUNDS = {
    "age": (0, 120), "arrival_hour": (0, 23), "systolic_bp": (40, 300),
    "diastolic_bp": (20, 200), "heart_rate": (20, 300),
    "respiratory_rate": (4, 60), "temperature": (30, 43), "spo2": (50, 100),
    "pain_score": (0, 10),
}

GENDER_MAP = {
    "F": 0, "f": 0, "female": 0,
    "M": 1, "m": 1, "male": 1,
}


# ---------------------------------------------------------------------------
# Data loading & outlier handling
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    """Load and lightly clean the raw triage CSV.

    - Drops rows whose encounter_id starts with 'VT' (synthetic validation rows).
    - Coerces numeric columns; leaves strings as-is.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    if "encounter_id" in df.columns:
        df = df[~df["encounter_id"].astype(str).str.startswith("VT")].copy()

    numeric_cols = [
        "age", "systolic_bp", "diastolic_bp", "heart_rate",
        "respiratory_rate", "temperature", "spo2", "pain_score",
        "arrival_hour", "acuity_target",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def classify_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-column IQR-based outlier bounds."""
    records = []
    for col in df.select_dtypes(include="number").columns:
        if col == TARGET_COL:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        records.append({
            "col":   col,
            "lower": q1 - 1.5 * iqr,
            "upper": q3 + 1.5 * iqr,
        })
    return pd.DataFrame(records)


def handle_outliers(df: pd.DataFrame, stats_tbl: pd.DataFrame) -> pd.DataFrame:
    """Replace values outside IQR bounds with NaN (artifact removal)."""
    d = df.copy()
    for _, row in stats_tbl.iterrows():
        col = row["col"]
        if col not in d.columns or col == TARGET_COL:
            continue
        mask = (d[col] < row["lower"]) | (d[col] > row["upper"])
        d.loc[mask, col] = np.nan
    return d


# ---------------------------------------------------------------------------
# Feature Engineering Helpers
# ---------------------------------------------------------------------------

def cyclical_encode(series: pd.Series, period: float = 24.0) -> tuple[pd.Series, pd.Series]:
    """Return (sin, cos) cyclical encoding for a periodic series."""
    angle = 2.0 * math.pi * series / period
    name = str(series.name) if series.name else "arrival"
    return np.sin(angle).rename(name + "_sin"), np.cos(angle).rename(name + "_cos")


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add shock_index, MAP, pulse_pressure, spo2_deficit, temp_deviation,
    resp_rate_ratio, boolean flags, qsofa_proxy, and NEWS2 proxy.
    """
    d = df.copy()

    # Continuous derived
    d["shock_index"]     = d["heart_rate"] / d["systolic_bp"].replace(0, np.nan)
    d["map"]             = (d["systolic_bp"] + 2.0 * d["diastolic_bp"]) / 3.0
    d["pulse_pressure"]  = d["systolic_bp"] - d["diastolic_bp"]
    d["spo2_deficit"]    = 100.0 - d["spo2"]
    d["temp_deviation"]  = (d["temperature"] - 37.0).abs()
    d["resp_rate_ratio"] = d["respiratory_rate"] / 20.0

    # Boolean flags
    d["is_tachycardic"] = (d["heart_rate"] > 100).astype(int)
    d["is_hypotensive"] = (d["systolic_bp"] < 90).astype(int)
    d["is_hypoxic"]     = (d["spo2"] < 92).astype(int)
    d["is_febrile"]     = (d["temperature"] >= 38.0).astype(int)

    # qSOFA proxy (0-3)
    rr_flag = (d["respiratory_rate"] >= 22).astype(int)
    d["qsofa_proxy"] = d["is_tachycardic"] + d["is_hypotensive"] + rr_flag

    # NEWS2 proxy
    def _news2_rr(x):
        if pd.isna(x): return 0
        if x <= 8 or x >= 25: return 3
        if 9 <= x <= 11: return 1
        if 21 <= x <= 24: return 2
        return 0

    def _news2_spo2(x):
        if pd.isna(x): return 0
        if x <= 91: return 3
        if x <= 93: return 2
        if x <= 95: return 1
        return 0

    def _news2_sbp(x):
        if pd.isna(x): return 0
        if x <= 90: return 3
        if x <= 100: return 2
        if x <= 110: return 1
        if x >= 220: return 3
        return 0

    def _news2_hr(x):
        if pd.isna(x): return 0
        if x <= 40 or x >= 131: return 3
        if x <= 50 or (111 <= x <= 130): return (1 if x <= 50 else 2)
        if 91 <= x <= 110: return 1
        return 0

    def _news2_temp(x):
        if pd.isna(x): return 0
        if x <= 35.0: return 3
        if x <= 36.0: return 1
        if x >= 39.1: return 2
        if x >= 38.1: return 1
        return 0

    d["news2_proxy"] = (
        d["respiratory_rate"].apply(_news2_rr)
        + d["spo2"].apply(_news2_spo2)
        + d["systolic_bp"].apply(_news2_sbp)
        + d["heart_rate"].apply(_news2_hr)
        + d["temperature"].apply(_news2_temp)
    )

    return d


def one_hot_age_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode age into five clinical buckets."""
    d = df.copy()
    cuts = pd.cut(d["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    for lbl in AGE_LABELS:
        d[lbl] = (cuts == lbl).astype(int)
    return d


def cross_fit_target_encode(
    df: pd.DataFrame,
    col: str,
    target: pd.Series,
    n_splits: int = 5,
) -> tuple[pd.Series, dict[str, float]]:
    """Cross-fitted target encoding for a categorical column — prevents leakage."""
    encoded = pd.Series(np.nan, index=df.index, name=col + "_enc")
    global_mean = float(target.mean())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for tr_idx, val_idx in kf.split(df):
        fold_map = (
            target.iloc[tr_idx]
            .groupby(df[col].iloc[tr_idx].values)
            .mean()
        )
        encoded.iloc[val_idx] = df[col].iloc[val_idx].map(fold_map).fillna(global_mean).values

    full_map = {str(k): float(v) for k, v in target.groupby(df[col].values).mean().to_dict().items()}
    return encoded.fillna(global_mean), full_map


def encode_gender(series: pd.Series) -> pd.Series:
    """Encode gender to F=0, M=1, unknown=-1."""
    return series.map(GENDER_MAP).fillna(-1).astype(int)


def apply_mice(
    df: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, Any]:
    """MICE imputation on continuous columns; median fallback if unavailable."""
    present = [c for c in cols if c in df.columns]
    d = df.copy()
    if _HAS_MICE:
        imp = IterativeImputer(
            max_iter=10, random_state=RANDOM_STATE,
            min_value=0, skip_complete=False, keep_empty_features=True,
        )
        d[present] = imp.fit_transform(d[present])
        return d, imp
    else:
        med = d[present].median()
        d[present] = d[present].fillna(med)
        return d, None


# ---------------------------------------------------------------------------
# Public Dataset Build API
# ---------------------------------------------------------------------------

def training_data(csv_path: str) -> pd.DataFrame:
    """Validate the dataset contract and retain labelled encounters."""
    df = load_data(csv_path)
    required = {*MICE_COLS, TARGET_COL, "arrival_hour", "chief_complaint", "gender"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing dataset columns: {', '.join(missing)}")
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    if set(df[TARGET_COL].unique()) != set(CLASSES):
        raise ValueError("Dataset must contain all four supported acuity labels.")
    return df


def transform_features(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Apply fitted transformations to validation or inference records."""
    frame = df.reindex(columns=[*MICE_COLS, "arrival_hour", "gender", "chief_complaint"]).copy()
    frame[MICE_COLS] = frame[MICE_COLS].apply(pd.to_numeric)
    bounds = meta["outlier_bounds"]
    frame = handle_outliers(frame, bounds)
    if meta["mice_imputer"] is not None:
        frame[MICE_COLS] = meta["mice_imputer"].transform(frame[MICE_COLS])
    frame[MICE_COLS] = frame[MICE_COLS].fillna(meta["medians"])
    frame = add_derived_features(frame)
    frame["arrival_sin"], frame["arrival_cos"] = cyclical_encode(
        pd.to_numeric(frame["arrival_hour"]), 24
    )
    frame["gender_enc"] = encode_gender(frame["gender"])
    frame["chief_complaint_enc"] = (
        frame["chief_complaint"].fillna("").str.strip().str.lower()
        .map(meta["cat_maps"]["chief_complaint"]).fillna(meta["target_mean"])
    )
    return one_hot_age_bucket(frame).reindex(columns=meta["feature_names"]).fillna(0)


def fit_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Fit preprocessing on a training partition and cross-fit complaint encoding."""
    df = df.copy().reset_index(drop=True)
    dm = df["chief_complaint"].fillna("").str.strip().str.lower()
    df["chief_complaint"] = dm
    y = df[TARGET_COL].astype(int)
    # Keep extreme readings inside the API validation ranges.
    bounds = pd.DataFrame(
        [(name, low, high) for name, (low, high) in INPUT_BOUNDS.items()],
        columns=["col", "lower", "upper"],
    )
    cleaned = handle_outliers(df, bounds)
    medians = cleaned[MICE_COLS].median().fillna(0)
    imputed, imputer = apply_mice(cleaned, MICE_COLS)
    derived = add_derived_features(imputed)
    features = [*MICE_COLS, *derived.columns.difference(df.columns),
                "arrival_sin", "arrival_cos", "chief_complaint_enc", "gender_enc",
                *AGE_LABELS]
    meta = {
        "feature_names": features, "outlier_bounds": bounds, "medians": medians,
        "mice_imputer": imputer, "target_mean": float(y.mean()),
        "cat_maps": {"chief_complaint": df.groupby("chief_complaint")[TARGET_COL].mean().to_dict()},
    }
    X = transform_features(df, meta)
    X["chief_complaint_enc"], _ = cross_fit_target_encode(df, "chief_complaint", y)
    return X, y, meta


def build_features(csv_path: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Prepare a dataset for standalone feature inspection."""
    return fit_features(training_data(csv_path))
