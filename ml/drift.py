"""
ml/drift.py
========================
Population drift monitoring via Population Stability Index (PSI).

Fits a quantile-sketch reference distribution on training data, then
computes per-feature PSI against any incoming batch.

PSI thresholds (industry standard):
    < 0.10  : no significant shift
    0.10-0.20: moderate shift — monitor
    > 0.20  : significant shift — [ALERT]

Public API
----------
fit_reference(X_train) -> None
check_drift(X_new, threshold=0.05) -> pd.DataFrame

Usage
-----
>>> from ml.drift import fit_reference, check_drift
>>> fit_reference(X_train)
>>> report = check_drift(X_new)
>>> print(report[report["psi"] > 0.10])
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_REF_PATH = "models/drift_reference.pkl"
_N_BINS           = 10
_PSI_WARN         = 0.10
_PSI_ALERT        = 0.20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _quantile_bins(series: np.ndarray, n_bins: int = _N_BINS) -> np.ndarray:
    """Compute quantile-based bin edges for a 1-D array.

    Args:
        series: 1-D numeric array (no NaN).
        n_bins: Number of bins.

    Returns:
        Array of bin edges (length n_bins+1), first and last extended to ±inf.

    Example:
        >>> edges = _quantile_bins(X_train["heart_rate"].values)
    """
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges     = np.percentile(series, quantiles)
    # Deduplicate edges (can happen with constant / highly discrete features)
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.array([-np.inf, np.inf])
    edges[0]  = -np.inf
    edges[-1] = np.inf
    return edges


def _compute_psi(
    ref: np.ndarray,
    new: np.ndarray,
    edges: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute Population Stability Index between ref and new distributions.

    PSI = sum[(P_new - P_ref) * ln(P_new / P_ref)]

    Args:
        ref:   Reference distribution array.
        new:   New distribution array.
        edges: Bin edges from reference fit.
        eps:   Small constant to avoid log(0).

    Returns:
        PSI float.

    Example:
        >>> psi = _compute_psi(ref_vals, new_vals, edges)
    """
    ref_cnt = np.histogram(ref, bins=edges)[0].astype(float)
    new_cnt = np.histogram(new, bins=edges)[0].astype(float)

    # Proportions
    ref_pct = (ref_cnt + eps) / (ref_cnt.sum() + eps * len(ref_cnt))
    new_pct = (new_cnt + eps) / (new_cnt.sum() + eps * len(new_cnt))

    psi = float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct + eps)))
    return max(psi, 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_reference(
    X_train: pd.DataFrame,
    ref_path: str = _DEFAULT_REF_PATH,
) -> None:
    """Fit and persist the reference distribution from training data.

    Computes quantile bin edges for every numeric feature in X_train
    and saves the sketch to disk.

    Args:
        X_train:  Training feature DataFrame (all numeric, no NaN).
        ref_path: Path to save the reference pickle.

    Example:
        >>> fit_reference(X_train, ref_path="models/drift_reference.pkl")
    """
    path = Path(ref_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    sketch: dict[str, dict] = {}
    for col in numeric_cols:
        vals = X_train[col].dropna().values
        if len(vals) == 0:
            continue
        edges = _quantile_bins(vals)
        sketch[col] = {
            "edges":  edges.tolist(),
            "values": vals.tolist(),   # stored for PSI computation
        }

    joblib.dump(sketch, path)
    print(f"[drift_monitor] Reference sketch saved: {path}  ({len(sketch)} features)")


def check_drift(
    X_new: pd.DataFrame,
    threshold: float = 0.05,
    ref_path: str = _DEFAULT_REF_PATH,
) -> pd.DataFrame:
    """Compute per-feature PSI between X_new and the reference distribution.

    Emits [ALERT] for any feature with PSI > 0.20.

    Args:
        X_new:     New batch DataFrame (same columns as X_train).
        threshold: PSI level below which drift is considered negligible.
                   (Used to annotate the 'status' column.)
        ref_path:  Path to the saved reference pickle from fit_reference().

    Returns:
        DataFrame with columns: feature, psi, status.
        status ∈ {'ok', 'monitor', 'alert'}.

    Example:
        >>> report = check_drift(X_new)
        >>> print(report[report["status"] == "alert"])
    """
    path = Path(ref_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference sketch not found at {ref_path}. "
            "Call fit_reference(X_train) first."
        )

    sketch = joblib.load(path)
    rows: list[dict] = []

    for col, ref_info in sketch.items():
        if col not in X_new.columns:
            warnings.warn(f"[warn] Feature '{col}' missing from X_new; PSI skipped.", stacklevel=2)
            continue

        new_vals = X_new[col].dropna().values
        ref_vals = np.asarray(ref_info["values"])
        edges    = np.asarray(ref_info["edges"])

        if len(new_vals) == 0:
            rows.append({"feature": col, "psi": float("nan"), "status": "no_data"})
            continue

        psi = _compute_psi(ref_vals, new_vals, edges)

        if psi > _PSI_ALERT:
            status = "alert"
            warnings.warn(
                f"[ALERT] Population shift detected: feature='{col}'  "
                f"PSI={psi:.4f} > {_PSI_ALERT}. "
                "Consider retraining or investigating data source.",
                stacklevel=2,
            )
        elif psi > _PSI_WARN:
            status = "monitor"
        elif psi > threshold:
            status = "monitor"
        else:
            status = "ok"

        rows.append({"feature": col, "psi": round(psi, 5), "status": status})

    df = (
        pd.DataFrame(rows)
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
    )
    n_alert   = int((df["status"] == "alert").sum())
    n_monitor = int((df["status"] == "monitor").sum())
    print(
        f"[drift_monitor] PSI check complete: "
        f"{n_alert} alert(s), {n_monitor} monitor(s), "
        f"{len(df) - n_alert - n_monitor} ok."
    )
    return df
