"""
ml/safety_metrics.py
=========================
Clinical safety metrics for ED-Triage acuity prediction.

Under-triage: predicting a lower acuity (higher number) than true.
Over-triage:  predicting a higher acuity (lower number) than true.
Critical miss: true Acuity-1 predicted as Acuity-3 or Acuity-4.

Public API
----------
compute_safety_metrics(y_true, y_pred) -> dict
compute_under_triage_by_class(y_true, y_pred) -> pd.DataFrame

Usage
-----
>>> from ml.safety_metrics import compute_safety_metrics
>>> m = compute_safety_metrics(y_true, y_pred)
>>> print(m["critical_miss_rate"])
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import LabelBinarizer

CLASSES = [1, 2, 3, 4]
_HIGH_ACUITY = {1, 2}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ece(y_true_bin: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) for a single class.

    Args:
        y_true_bin: Binary array (0/1) for one class.
        y_proba:    Predicted probabilities for that class.
        n_bins:     Number of calibration bins.

    Returns:
        ECE float.

    Example:
        >>> err = _ece(y_bin[:, 0], y_proba[:, 0])
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n = len(y_true_bin)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        acc  = float(y_true_bin[mask].mean())
        conf = float(y_proba[mask].mean())
        ece_val += (mask.sum() / n) * abs(acc - conf)
    return ece_val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_safety_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray | None = None,
) -> dict:
    """Compute clinical safety and ML performance metrics.

    Args:
        y_true:  True acuity labels (1-4, int).
        y_pred:  Predicted acuity labels (1-4, int).
        y_proba: Optional predicted probability matrix (N × 4).
                 Required for ECE and Brier score.

    Returns:
        Dict with keys:
            under_triage_rate_overall   – fraction of high-acuity cases
                                          predicted as lower acuity
            under_triage_rate_acuity1   – fraction of true-1 predicted >1
            under_triage_rate_acuity2   – fraction of true-2 predicted >2
            over_triage_rate_overall    – fraction predicted higher than true
            high_acuity_sensitivity     – recall for classes {1,2} combined
            high_acuity_specificity     – TN/(TN+FP) for {1,2} vs {3,4}
            critical_miss_rate          – true Acuity-1 predicted as 3 or 4
            macro_f1                    – sklearn macro F1
            weighted_f1                 – sklearn weighted F1
            ece_macro                   – macro-averaged ECE (requires proba)
            brier_macro                 – macro-averaged Brier (requires proba)

    Example:
        >>> m = compute_safety_metrics(y_true, y_pred, y_proba)
        >>> print(m["critical_miss_rate"])
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    ha_mask = np.isin(yt, list(_HIGH_ACUITY))
    n_ha    = int(ha_mask.sum())

    # Under-triage (predicted acuity > true acuity = less urgent)
    under_mask  = yp > yt
    under_all   = float(under_mask[ha_mask].mean()) if n_ha > 0 else 0.0

    a1_mask = yt == 1
    under1  = float((yp[a1_mask] > 1).mean()) if a1_mask.sum() > 0 else 0.0
    a2_mask = yt == 2
    under2  = float((yp[a2_mask] > 2).mean()) if a2_mask.sum() > 0 else 0.0

    # Over-triage (predicted acuity < true acuity = more urgent)
    over_mask = yp < yt
    over_all  = float(over_mask.mean())

    # High-acuity sensitivity/specificity (binary: {1,2} vs {3,4})
    ha_true  = ha_mask.astype(int)
    ha_pred  = np.isin(yp, list(_HIGH_ACUITY)).astype(int)
    tp = int(((ha_pred == 1) & (ha_true == 1)).sum())
    fn = int(((ha_pred == 0) & (ha_true == 1)).sum())
    tn = int(((ha_pred == 0) & (ha_true == 0)).sum())
    fp = int(((ha_pred == 1) & (ha_true == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Critical miss rate: true Acuity-1 predicted as 3 or 4
    crit_mask      = yt == 1
    critical_miss  = float((yp[crit_mask] >= 3).mean()) if crit_mask.sum() > 0 else 0.0

    # F1 scores
    macro_f1    = float(f1_score(yt, yp, average="macro", labels=CLASSES, zero_division=0))
    weighted_f1 = float(f1_score(yt, yp, average="weighted", labels=CLASSES, zero_division=0))

    # ECE + Brier (optional — require probabilities)
    ece_macro   = float("nan")
    brier_macro = float("nan")
    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=float)
        lb = LabelBinarizer()
        lb.fit(CLASSES)
        y_bin = lb.transform(yt)
        ece_vals    = [_ece(y_bin[:, i], proba[:, i]) for i in range(4)]
        brier_vals  = [brier_score_loss(y_bin[:, i], proba[:, i]) for i in range(4)]
        ece_macro   = float(np.mean(ece_vals))
        brier_macro = float(np.mean(brier_vals))

    return {
        "under_triage_rate_overall": under_all,
        "under_triage_rate_acuity1": under1,
        "under_triage_rate_acuity2": under2,
        "over_triage_rate_overall":  over_all,
        "high_acuity_sensitivity":   sensitivity,
        "high_acuity_specificity":   specificity,
        "critical_miss_rate":        critical_miss,
        "macro_f1":                  macro_f1,
        "weighted_f1":               weighted_f1,
        "ece_macro":                 ece_macro,
        "brier_macro":               brier_macro,
    }


def compute_under_triage_by_class(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Compute under-triage statistics broken down by true acuity class.

    Under-triage for class k = fraction of true-k cases predicted as >k.

    Args:
        y_true: True acuity labels (1-4, int).
        y_pred: Predicted acuity labels (1-4, int).

    Returns:
        DataFrame with columns:
            true_acuity, n_cases, n_under_triaged, under_triage_rate.

    Example:
        >>> df = compute_under_triage_by_class(y_true, y_pred)
        >>> print(df.to_string())
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    rows = []
    for cls in CLASSES:
        mask    = yt == cls
        n       = int(mask.sum())
        n_under = int((yp[mask] > cls).sum()) if n > 0 else 0
        rows.append({
            "true_acuity":       cls,
            "n_cases":           n,
            "n_under_triaged":   n_under,
            "under_triage_rate": round(n_under / n, 4) if n > 0 else float("nan"),
        })
    return pd.DataFrame(rows)
