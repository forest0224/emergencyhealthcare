"""
ml/fairness_audit.py
=========================
Subgroup fairness audit for the ED-Triage acuity prediction model.

Computes recall, precision, F1, and disparate-impact ratio per subgroup.
Emits [WARN] if any subgroup's Acuity-1/2 sensitivity falls >10% below
the overall population sensitivity.

Public API
----------
audit_fairness(X, y, y_pred,
               protected_cols=('age_bucket','gender')) -> pd.DataFrame

Usage
-----
>>> from ml.fairness_audit import audit_fairness
>>> report = audit_fairness(X, y, y_pred)
>>> print(report.to_string())
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

CLASSES      = [1, 2, 3, 4]
_HIGH_ACUITY = {1, 2}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_name: str,
    group_val: str,
) -> dict:
    """Compute metrics for one subgroup.

    Args:
        y_true:     True acuity labels for this subgroup.
        y_pred:     Predicted acuity labels for this subgroup.
        group_name: Protected attribute name (e.g. 'gender').
        group_val:  Specific group value (e.g. 'M').

    Returns:
        Dict of metric_name → value.

    Example:
        >>> m = _subgroup_metrics(yt, yp, "gender", "M")
    """
    n = len(y_true)
    if n == 0:
        return {}

    ha_true = np.isin(y_true, list(_HIGH_ACUITY)).astype(int)
    ha_pred = np.isin(y_pred, list(_HIGH_ACUITY)).astype(int)

    tp = int(((ha_pred == 1) & (ha_true == 1)).sum())
    fn = int(((ha_pred == 0) & (ha_true == 1)).sum())
    fp = int(((ha_pred == 1) & (ha_true == 0)).sum())

    ha_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    ha_precision   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    # Acuity-1 recall
    a1_mask = y_true == 1
    a1_rec  = float((y_pred[a1_mask] == 1).mean()) if a1_mask.sum() > 0 else float("nan")

    macro_f1 = float(f1_score(y_true, y_pred, average="macro",
                               labels=[c for c in CLASSES if c in np.unique(y_true)],
                               zero_division=0))

    return {
        "subgroup":            f"{group_name}={group_val}",
        "n":                   n,
        "ha_sensitivity":      ha_sensitivity,
        "ha_precision":        ha_precision,
        "acuity1_recall":      a1_rec,
        "macro_f1":            macro_f1,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def audit_fairness(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    y_pred: np.ndarray | pd.Series,
    protected_cols: Sequence[str] = ("age_bucket", "gender"),
) -> pd.DataFrame:
    """Compute subgroup-level fairness metrics.

    For each unique value of each protected column, compute recall, precision,
    F1, and disparate-impact ratio relative to the overall population.

    Emits [WARN] if any subgroup's Acuity-1/2 sensitivity is more than 10%
    below the overall sensitivity.

    Args:
        X:              Feature DataFrame that contains the protected columns
                        (or columns that can be mapped back to group labels).
                        Also accepts any DataFrame with those columns present.
        y:              True acuity labels (1-4).
        y_pred:         Predicted acuity labels (1-4).
        protected_cols: Columns to audit. Defaults to ('age_bucket','gender').

    Returns:
        DataFrame with one row per subgroup plus an 'overall' row, columns:
            subgroup, n, ha_sensitivity, ha_precision, acuity1_recall,
            macro_f1, disparate_impact_ratio.

    Example:
        >>> report = audit_fairness(X_test, y_test, model.predict(X_test))
        >>> print(report[["subgroup", "ha_sensitivity", "disparate_impact_ratio"]])
    """
    yt   = np.asarray(y, dtype=int)
    yp   = np.asarray(y_pred, dtype=int)
    rows: list[dict] = []

    # Overall metrics (baseline)
    overall = _subgroup_metrics(yt, yp, "overall", "all")
    overall["disparate_impact_ratio"] = 1.0
    rows.append(overall)
    overall_ha_sens = overall.get("ha_sensitivity", float("nan"))

    for col in protected_cols:
        if col not in X.columns:
            warnings.warn(
                f"[warn] Protected column '{col}' not found in X; skipping.",
                stacklevel=2,
            )
            continue

        col_vals = X[col].astype(str)
        for group_val in sorted(col_vals.unique()):
            mask  = (col_vals == group_val).values
            if mask.sum() == 0:
                continue
            sg = _subgroup_metrics(yt[mask], yp[mask], col, group_val)
            if not sg:
                continue

            # Disparate impact ratio: subgroup ha_sensitivity / overall
            if not np.isnan(overall_ha_sens) and overall_ha_sens > 0:
                sg["disparate_impact_ratio"] = round(
                    sg.get("ha_sensitivity", 0.0) / overall_ha_sens, 4
                )
            else:
                sg["disparate_impact_ratio"] = float("nan")

            rows.append(sg)

            # Emit warning if sensitivity gap > 10%
            sg_sens = sg.get("ha_sensitivity", float("nan"))
            if (
                not np.isnan(sg_sens)
                and not np.isnan(overall_ha_sens)
                and (overall_ha_sens - sg_sens) > 0.10
            ):
                warnings.warn(
                    f"[WARN] Subgroup {col}={group_val}: Acuity-1/2 sensitivity "
                    f"= {sg_sens:.3f}, which is "
                    f"{overall_ha_sens - sg_sens:.3f} below overall "
                    f"({overall_ha_sens:.3f}). Review for disparate impact.",
                    stacklevel=2,
                )

    df = pd.DataFrame(rows)
    return df
