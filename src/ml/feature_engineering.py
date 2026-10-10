"""
src/ml/feature_engineering.py
===============================
Feature engineering for the ED-Triage ML pipeline.

Imports read-only helpers from ed_triage_eda_pipeline; never modifies them.

Public API
----------
build_features(csv_path: str) -> tuple[pd.DataFrame, pd.Series, dict]

Usage
-----
>>> X, y, meta = build_features("master_emergency_triage_dataset.csv")
>>> print(X.shape, y.value_counts().to_dict())
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
        "[warn] sklearn IterativeImputer unavailable; falling back to "
        "median imputation.",
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Read-only imports from the existing pipeline (never modified)
# ---------------------------------------------------------------------------
from ed_triage_eda_pipeline import (  # noqa: E402
    load_data,
    classify_outliers,
    handle_outliers,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET_COL   = "acuity_target"

# Continuous vitals that go into MICE
_MICE_COLS = [
    "age", "systolic_bp", "diastolic_bp", "heart_rate",
    "respiratory_rate", "temperature", "spo2", "pain_score",
]

# Age bucket boundaries and names
_AGE_BINS   = [0, 17, 39, 64, 79, 120]
_AGE_LABELS = ["age_0_17", "age_18_39", "age_40_64", "age_65_79", "age_80plus"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _cyclical_encode(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Return (sin, cos) cyclical encoding for a periodic series.

    Args:
        series: Numeric series (e.g. arrival_hour 0-23).
        period: Full-cycle length (24 for hours).

    Returns:
        Tuple (sin_series, cos_series).

    Example:
        >>> s, c = _cyclical_encode(df["arrival_hour"], 24)
    """
    angle = 2.0 * math.pi * series / period
    return np.sin(angle).rename(series.name + "_sin"), np.cos(angle).rename(series.name + "_cos")


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add shock_index, MAP, pulse_pressure, spo2_deficit, temp_deviation,
    resp_rate_ratio, boolean flags, qsofa_proxy, and NEWS2 proxy.

    NEWS2 weights used (sourced from Royal College of Physicians NEWS2 2017):
        RR:  ≤8→3, 9-11→1, 12-20→0, 21-24→2, ≥25→3
        SpO2:≥96→0, 94-95→1, 92-93→2, ≤91→3
        SBP: ≤90→3, 91-100→2, 101-110→1, 111-219→0, ≥220→3
        HR:  ≤40→3, 41-50→1, 51-90→0, 91-110→1, 111-130→2, ≥131→3
        Temp:≤35.0→3, 35.1-36.0→1, 36.1-38.0→0, 38.1-39.0→1, ≥39.1→2

    Args:
        df: DataFrame with vitals columns (NaN safe).

    Returns:
        df copy with derived columns appended.

    Example:
        >>> df = _add_derived_features(df)
    """
    d = df.copy()

    # --- Continuous derived ---
    d["shock_index"]      = d["heart_rate"] / d["systolic_bp"].replace(0, np.nan)
    d["map"]              = (d["systolic_bp"] + 2.0 * d["diastolic_bp"]) / 3.0
    d["pulse_pressure"]   = d["systolic_bp"] - d["diastolic_bp"]
    d["spo2_deficit"]     = 100.0 - d["spo2"]
    d["temp_deviation"]   = (d["temperature"] - 37.0).abs()
    d["resp_rate_ratio"]  = d["respiratory_rate"] / 20.0

    # --- Boolean flags ---
    d["is_tachycardic"]  = (d["heart_rate"] > 100).astype(int)
    d["is_hypotensive"]  = (d["systolic_bp"] < 90).astype(int)
    d["is_hypoxic"]      = (d["spo2"] < 92).astype(int)
    d["is_febrile"]      = (d["temperature"] >= 38.0).astype(int)

    # --- qSOFA proxy (0-3) ---
    rr_flag = (d["respiratory_rate"] >= 22).astype(int)
    d["qsofa_proxy"] = d["is_tachycardic"] + d["is_hypotensive"] + rr_flag

    # --- NEWS2 proxy ---
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


def _one_hot_age_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode age into five clinical buckets.

    Buckets: 0-17 (pediatric), 18-39, 40-64, 65-79, 80+ (elderly).

    Args:
        df: DataFrame with 'age' column.

    Returns:
        df copy with age_0_17, age_18_39, age_40_64, age_65_79, age_80plus
        columns added (int 0/1).

    Example:
        >>> df = _one_hot_age_bucket(df)
    """
    d = df.copy()
    cuts = pd.cut(d["age"], bins=_AGE_BINS, labels=_AGE_LABELS, right=True)
    for lbl in _AGE_LABELS:
        d[lbl] = (cuts == lbl).astype(int)
    return d


def _cross_fit_target_encode(
    df: pd.DataFrame,
    col: str,
    target: pd.Series,
    n_splits: int = 5,
) -> tuple[pd.Series, dict]:
    """Cross-fitted target encoding for a categorical column — no leakage.

    Encodes using out-of-fold target means; unseen values map to global mean.

    Args:
        df:       Input DataFrame.
        col:      Categorical column name.
        target:   Integer target Series (same index as df).
        n_splits: Number of CV folds.

    Returns:
        Tuple of (encoded_series: float, encoding_map: {category: mean}).

    Example:
        >>> enc, mp = _cross_fit_target_encode(df, "chief_complaint", y)
    """
    encoded    = pd.Series(np.nan, index=df.index, name=col + "_enc")
    global_mean = float(target.mean())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for tr_idx, val_idx in kf.split(df):
        fold_map = (
            target.iloc[tr_idx]
            .groupby(df[col].iloc[tr_idx].values)
            .mean()
        )
        encoded.iloc[val_idx] = df[col].iloc[val_idx].map(fold_map).fillna(global_mean).values

    # Full-data map for inference-time use
    full_map = target.groupby(df[col].values).mean().to_dict()
    return encoded.fillna(global_mean), full_map


def _encode_gender(series: pd.Series) -> pd.Series:
    """Encode gender to F=0, M=1, unknown=-1.

    Args:
        series: Raw gender Series (strings).

    Returns:
        Integer-encoded Series.

    Example:
        >>> enc_g = _encode_gender(df["gender"])
    """
    mapping = {"F": 0, "f": 0, "female": 0,
               "M": 1, "m": 1, "male":   1}
    return series.map(mapping).fillna(-1).astype(int)


def _apply_mice(
    df: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, Any]:
    """MICE imputation on continuous columns; median fallback if unavailable.

    Args:
        df:   DataFrame (may contain NaN).
        cols: Columns to impute.

    Returns:
        Tuple (imputed_df, fitted_imputer_or_None).

    Example:
        >>> df, imp = _apply_mice(df, _MICE_COLS)
    """
    present = [c for c in cols if c in df.columns]
    d = df.copy()
    if _HAS_MICE:
        imp = IterativeImputer(
            max_iter=10, random_state=RANDOM_STATE,
            min_value=0, skip_complete=True,
        )
        d[present] = imp.fit_transform(d[present])
        return d, imp
    else:
        med = d[present].median()
        d[present] = d[present].fillna(med)
        return d, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    csv_path: str,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Build the full feature matrix from the raw CSV.

    Pipeline:
        1. load_data → drops VT#### rows, coerces types.
        2. classify_outliers / handle_outliers → artifact NaN.
        3. MICE imputation on continuous vitals.
        4. Derived clinical features (shock_index, MAP, etc.).
        5. Cyclical arrival_hour encoding.
        6. Cross-fitted target encoding of chief_complaint (5-fold).
        7. Gender numeric encoding.
        8. Age bucket one-hot encoding.
        9. Drop non-feature columns; fill residual NaN with 0.

    Args:
        csv_path: Path to master_emergency_triage_dataset.csv.

    Returns:
        X    – pd.DataFrame, all numeric, no NaN.
        y    – pd.Series[int], acuity_target (1-4).
        meta – dict with keys:
                 feature_names    : list[str]
                 cat_maps         : {"chief_complaint": {str: float}}
                 scaled_cols      : list[str]
                 mice_imputer     : fitted IterativeImputer or None

    Example:
        >>> X, y, meta = build_features("master_emergency_triage_dataset.csv")
        >>> X.shape
        (N, 30+)
    """
    # 1. Load
    df = load_data(csv_path)

    # 2. Outlier handling
    stats_tbl = classify_outliers(df)
    df = handle_outliers(df, stats_tbl)

    # 3. MICE on continuous vitals
    df, mice_imp = _apply_mice(df, _MICE_COLS)

    # 4. Derived clinical features
    df = _add_derived_features(df)

    # 5. Cyclical arrival hour
    if "arrival_hour" in df.columns:
        s, c = _cyclical_encode(df["arrival_hour"], 24)
        df["arrival_sin"] = s.values
        df["arrival_cos"] = c.values

    # Extract target before encoding (target encoding needs y)
    y = df[TARGET_COL].astype(int).copy()

    # 6. Cross-fitted target encoding for chief_complaint
    cat_maps: dict[str, dict] = {}
    if "chief_complaint" in df.columns:
        enc_series, cc_map = _cross_fit_target_encode(
            df, "chief_complaint", y
        )
        df["chief_complaint_enc"] = enc_series.values
        cat_maps["chief_complaint"] = cc_map

    # 7. Gender
    if "gender" in df.columns:
        df["gender_enc"] = _encode_gender(df["gender"])

    # 8. Age buckets
    df = _one_hot_age_bucket(df)

    # 9. Assemble X — drop non-feature columns
    _DROP = {
        TARGET_COL, "encounter_id", "chief_complaint",
        "gender", "arrival_hour",
    }
    # Also drop any *_scaled columns from the EDA pipeline's scaler
    # and any remaining object columns
    feature_cols = [
        c for c in df.columns
        if c not in _DROP
        and not c.endswith("_scaled")
        and df[c].dtype != object
    ]

    X = df[feature_cols].copy().fillna(0)

    # Identify which cols were scaled (none in this flow — we keep raw + derived)
    scaled_cols: list[str] = []

    meta: dict[str, Any] = {
        "feature_names":  feature_cols,
        "cat_maps":       cat_maps,
        "scaled_cols":    scaled_cols,
        "mice_imputer":   mice_imp,
    }

    print(
        f"[feature_engineering] X shape: {X.shape}  |  "
        f"target dist: {y.value_counts().sort_index().to_dict()}"
    )
    return X, y, meta
