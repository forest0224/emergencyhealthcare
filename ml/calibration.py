"""
ml/calibration.py
======================
Post-hoc probability calibration for the ED-Triage model.

Compares ECE before and after calibration and returns the better model.

Public API
----------
calibrate(model, X_val, y_val, method='isotonic') -> model

Usage
-----
>>> from ml.calibration import calibrate
>>> cal_model = calibrate(best_model, X_val, y_val)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelBinarizer

from ml.safety_metrics import binary_ece

CLASSES      = [1, 2, 3, 4]
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ece_multiclass(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Macro-averaged Expected Calibration Error across all classes.

    Args:
        y_true:  True integer labels (1-4).
        y_proba: Predicted probability matrix (N × 4).
        n_bins:  Number of calibration bins.

    Returns:
        Macro-averaged ECE float.

    Example:
        >>> ece = _ece_multiclass(y_true, proba)
    """
    lb = LabelBinarizer()
    lb.fit(CLASSES)
    y_bin = lb.transform(y_true)

    return float(np.mean([
        binary_ece(y_bin[:, i], y_proba[:, i], n_bins)
        for i in range(y_bin.shape[1])
    ]))


def _get_proba(model, X: np.ndarray) -> np.ndarray:
    """Get a (N, 4) probability matrix, aligned to CLASSES.

    Args:
        model: Fitted sklearn-compatible estimator.
        X:     Feature array.

    Returns:
        Probability array of shape (N, 4).

    Example:
        >>> proba = _get_proba(model, X_val)
    """
    raw = model.predict_proba(X)
    cls_list = list(model.classes_)
    aligned  = np.zeros((len(X), 4), dtype=float)
    zero_based = set(cls_list) == {0, 1, 2, 3}
    for i, cls in enumerate(cls_list):
        orig = int(cls) + int(zero_based)
        if 1 <= orig <= 4:
            aligned[:, orig - 1] = raw[:, i]
    return aligned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calibrate(
    model,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    method: str = "isotonic",
    *,
    always_fit: bool = False,
) -> object:
    """Calibrate model probabilities and return the better-calibrated version.

    Fits a CalibratedClassifierCV wrapper on (X_val, y_val) using the
    specified method ('isotonic' or 'sigmoid'), then compares ECE before
    and after.  Returns whichever has lower ECE.

    Args:
        model:   Fitted sklearn-compatible classifier.
        X_val:   Validation feature array / DataFrame.
        y_val:   Validation labels in the fitted estimator's encoding.
        method:  Calibration method: 'isotonic' (default) or 'sigmoid'.

    Returns:
        The model with lower ECE (calibrated or original).

    Example:
        >>> from ml.calibration import calibrate
        >>> cal_model = calibrate(best_model, X_val, y_val)
        >>> print("calibrated" if cal_model is not best_model else "original")
    """
    X = np.asarray(X_val, dtype=np.float32)
    y = np.asarray(y_val, dtype=int)

    evaluation_labels = y + 1 if set(model.classes_) == {0, 1, 2, 3} else y

    # ---- ECE before calibration ----
    proba_before = _get_proba(model, X)
    ece_before   = _ece_multiclass(evaluation_labels, proba_before)
    print(f"[calibration] ECE before ({method}): {ece_before:.5f}")

    # ---- Fit calibrated wrapper ----
    # CalibratedClassifierCV needs labels in 0-based for some backends;
    # we wrap with cv='prefit' since the model is already fitted.
    try:
        cal = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv="prefit",
        )
        cal.fit(X, y)
    except Exception as exc:
        warnings.warn(
            f"[warn] Calibration failed ({exc}); returning original model.",
            stacklevel=2,
        )
        return model

    # ---- ECE after calibration ----
    proba_after = _get_proba(cal, X)
    ece_after   = _ece_multiclass(evaluation_labels, proba_after)
    print(f"[calibration] ECE after  ({method}): {ece_after:.5f}")

    if always_fit or ece_after < ece_before:
        print(
            f"[calibration] Calibrated model selected "
            f"(calibration-set ECE change {ece_after - ece_before:+.5f})."
        )
        return cal
    else:
        print(
            f"[calibration] Original model retained "
            f"(calibration did not improve ECE)."
        )
        return model
