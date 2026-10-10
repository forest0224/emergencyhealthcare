"""
src/ml/inference_api.py
========================
Production inference wrapper for the trained ED-Triage model.

Loads the best_model.pkl + feature_meta.pkl saved by tabular_models.py
and applies the SAME feature engineering pipeline at inference time.

Public API
----------
class TriageModel
    predict_proba(patient_dict) -> dict
    explain(patient_dict, top_k=5) -> dict

Usage
-----
>>> model = TriageModel("ml_runs")
>>> out = model.predict_proba({"age": 65, "heart_rate": 110, ...})
>>> print(out["predicted"], out["confidence"])
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional: SHAP
# ---------------------------------------------------------------------------
try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    warnings.warn("[warn] shap not installed; explain() will return empty dict.", stacklevel=2)

# ---------------------------------------------------------------------------
# Constants (must match feature_engineering.py)
# ---------------------------------------------------------------------------
CLASSES      = [1, 2, 3, 4]
RANDOM_STATE = 42
_MICE_COLS   = [
    "age", "systolic_bp", "diastolic_bp", "heart_rate",
    "respiratory_rate", "temperature", "spo2", "pain_score",
]
_AGE_BINS   = [0, 17, 39, 64, 79, 120]
_AGE_LABELS = ["age_0_17", "age_18_39", "age_40_64", "age_65_79", "age_80plus"]

_GENDER_MAP = {
    "F": 0, "f": 0, "female": 0,
    "M": 1, "m": 1, "male": 1,
}


# ---------------------------------------------------------------------------
# Feature construction (mirrors feature_engineering.py, single-row safe)
# ---------------------------------------------------------------------------

def _build_single_row(patient: dict, meta: dict) -> tuple[pd.DataFrame, list[str]]:
    """Build one-row feature DataFrame from a patient dict.

    Applies the same transformations as build_features():
        derived vitals, cyclical encoding, target encoding,
        gender encoding, age bucket one-hot.

    Args:
        patient: Raw patient dict with column-name keys.
        meta:    Metadata dict from training (feature_names, cat_maps, etc.).

    Returns:
        Tuple (row_df, imputed_fields) where imputed_fields lists
        any columns that were missing in patient and filled.

    Example:
        >>> row, imp = _build_single_row({"age": 45, "heart_rate": 90}, meta)
    """
    d: dict[str, Any] = {k: v for k, v in patient.items()}

    # Track which fields were missing (will be imputed)
    imputed_fields: list[str] = [
        c for c in _MICE_COLS if (c not in d or d[c] is None)
    ]

    # Fill missing continuous vitals with median-like defaults (0 triggers MICE)
    for c in _MICE_COLS:
        if c not in d or d[c] is None:
            d[c] = np.nan

    # --- Derived vitals ---
    hr  = d.get("heart_rate",  np.nan)
    sbp = d.get("systolic_bp", np.nan)
    dbp = d.get("diastolic_bp", np.nan)
    rr  = d.get("respiratory_rate", np.nan)
    spo2 = d.get("spo2", np.nan)
    temp = d.get("temperature", np.nan)

    def _safe_div(a, b):
        try:
            if b and b != 0:
                return float(a) / float(b)
        except Exception:
            pass
        return np.nan

    d["shock_index"]     = _safe_div(hr, sbp)
    d["map"]             = ((sbp or 0) + 2 * (dbp or 0)) / 3.0
    d["pulse_pressure"]  = (sbp or 0) - (dbp or 0)
    d["spo2_deficit"]    = 100.0 - (spo2 if spo2 is not None else np.nan)
    d["temp_deviation"]  = abs((temp or 37.0) - 37.0)
    d["resp_rate_ratio"] = (rr or 0) / 20.0

    d["is_tachycardic"] = int((hr or 0) > 100)
    d["is_hypotensive"] = int((sbp or 0) < 90)
    d["is_hypoxic"]     = int((spo2 or 100) < 92)
    d["is_febrile"]     = int((temp or 37.0) >= 38.0)

    rr_flag = int((rr or 0) >= 22)
    d["qsofa_proxy"] = d["is_tachycardic"] + d["is_hypotensive"] + rr_flag

    # NEWS2 proxy
    def _n2_rr(x):
        if x is None or np.isnan(x): return 0
        if x <= 8 or x >= 25: return 3
        if 9 <= x <= 11: return 1
        if 21 <= x <= 24: return 2
        return 0
    def _n2_spo2(x):
        if x is None or np.isnan(x): return 0
        if x <= 91: return 3
        if x <= 93: return 2
        if x <= 95: return 1
        return 0
    def _n2_sbp(x):
        if x is None or np.isnan(x): return 0
        if x <= 90: return 3
        if x <= 100: return 2
        if x <= 110: return 1
        if x >= 220: return 3
        return 0
    def _n2_hr(x):
        if x is None or np.isnan(x): return 0
        if x <= 40 or x >= 131: return 3
        if x <= 50 or (111 <= x <= 130): return (1 if x <= 50 else 2)
        if 91 <= x <= 110: return 1
        return 0
    def _n2_temp(x):
        if x is None or np.isnan(x): return 0
        if x <= 35.0: return 3
        if x <= 36.0: return 1
        if x >= 39.1: return 2
        if x >= 38.1: return 1
        return 0

    d["news2_proxy"] = (
        _n2_rr(rr) + _n2_spo2(spo2) + _n2_sbp(sbp) + _n2_hr(hr) + _n2_temp(temp)
    )

    # Cyclical arrival hour
    ah = d.get("arrival_hour", 0) or 0
    d["arrival_sin"] = math.sin(2 * math.pi * ah / 24)
    d["arrival_cos"] = math.cos(2 * math.pi * ah / 24)

    # Chief complaint target encoding
    cc_map = meta.get("cat_maps", {}).get("chief_complaint", {})
    cc_val = d.get("chief_complaint", "")
    global_mean = float(np.mean(list(cc_map.values()))) if cc_map else 2.5
    d["chief_complaint_enc"] = float(cc_map.get(cc_val, global_mean))

    # Gender
    gender_raw = str(d.get("gender", "")).strip()
    d["gender_enc"] = _GENDER_MAP.get(gender_raw, -1)

    # Age buckets (one-hot)
    age_val = d.get("age", np.nan)
    try:
        age_val = float(age_val) if age_val is not None else np.nan
    except (ValueError, TypeError):
        age_val = np.nan
    for lbl in _AGE_LABELS:
        d[lbl] = 0
    if not np.isnan(age_val):
        for i in range(len(_AGE_BINS) - 1):
            if _AGE_BINS[i] <= age_val <= _AGE_BINS[i + 1]:
                d[_AGE_LABELS[i]] = 1
                break

    # Build row aligned to training feature order
    feature_names: list[str] = meta["feature_names"]
    row_vals = {f: d.get(f, 0.0) for f in feature_names}
    row_df = pd.DataFrame([row_vals], columns=feature_names)

    # Fill any remaining NaN with 0
    row_df = row_df.fillna(0.0)

    return row_df, imputed_fields


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TriageModel:
    """Production inference wrapper for the trained ED-Triage model.

    Loads best_model.pkl and feature_meta.pkl from model_dir and applies
    the identical feature engineering pipeline used at training time.

    Args:
        model_dir: Path to directory containing best_model.pkl and
                   feature_meta.pkl (default "ml_runs").

    Example:
        >>> tm = TriageModel("ml_runs")
        >>> result = tm.predict_proba({"age": 65, "heart_rate": 115,
        ...                            "systolic_bp": 85, "spo2": 94})
    """

    def __init__(self, model_dir: str = "ml_runs") -> None:
        """Load model and metadata from model_dir.

        Args:
            model_dir: Directory produced by tabular_models.train_and_evaluate.

        Example:
            >>> tm = TriageModel("ml_runs")
        """
        mdir = Path(model_dir)
        model_path = mdir / "best_model.pkl"
        meta_path  = mdir / "feature_meta.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"best_model.pkl not found in {model_dir}. "
                "Run `make train` first."
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"feature_meta.pkl not found in {model_dir}. "
                "Run `make train` first."
            )

        self._model = joblib.load(model_path)
        self._meta  = joblib.load(meta_path)
        self._label_imap: dict[int, int] = self._meta.get("label_imap", {i: i+1 for i in range(4)})
        self._explainer: Any = None  # lazy-init

        print(f"[TriageModel] Loaded {self._meta.get('best_model', 'model')} from {model_dir}")

    # ------------------------------------------------------------------ #

    def predict_proba(self, patient_dict: dict) -> dict:
        """Run inference on a single patient dict.

        Args:
            patient_dict: Dict with keys matching CSV columns.
                          Missing vitals are imputed (flagged in output).

        Returns:
            Dict with keys:
                proba          – list[float] length 4, P(Acuity=1..4)
                predicted      – int, most likely acuity (1-4)
                confidence     – float, max probability
                imputed_fields – list[str] of fields filled by imputation
                feature_vector – dict of feature_name → float

        Example:
            >>> out = tm.predict_proba({"age": 70, "systolic_bp": 80})
            >>> print(out["predicted"], out["confidence"])
        """
        row_df, imputed_fields = _build_single_row(patient_dict, self._meta)
        X = row_df.values.astype(np.float32)

        raw_proba = self._model.predict_proba(X)   # shape (1, n_classes)
        raw_pred  = self._model.predict(X)

        # Align to CLASSES order using label_imap
        classes_in = list(self._model.classes_)
        proba_aligned = np.zeros(4, dtype=float)
        for col_i, cls_i in enumerate(classes_in):
            orig = self._label_imap.get(int(cls_i), int(cls_i))
            if 1 <= orig <= 4:
                proba_aligned[orig - 1] = float(raw_proba[0, col_i])

        predicted_raw = int(raw_pred[0])
        predicted     = self._label_imap.get(predicted_raw, predicted_raw)
        confidence    = float(proba_aligned.max())

        return {
            "proba":          proba_aligned.tolist(),
            "predicted":      predicted,
            "confidence":     confidence,
            "imputed_fields": imputed_fields,
            "feature_vector": row_df.iloc[0].to_dict(),
        }

    # ------------------------------------------------------------------ #

    def explain(self, patient_dict: dict, top_k: int = 5) -> dict:
        """Return a local SHAP explanation for a single patient.

        Requires the ``shap`` package.  If unavailable, returns empty dict
        with a warning.

        Args:
            patient_dict: Same format as predict_proba().
            top_k:        Number of top features to return per class.

        Returns:
            Dict mapping acuity class (str "1".."4") to a list of
            (feature_name, shap_value) pairs, sorted by abs(shap) desc.

        Example:
            >>> exp = tm.explain({"age": 70, "heart_rate": 120}, top_k=5)
            >>> print(exp["1"])
        """
        if not _HAS_SHAP:
            warnings.warn("[warn] shap not installed; returning empty explanation.", stacklevel=2)
            return {}

        row_df, _ = _build_single_row(patient_dict, self._meta)
        X = row_df.values.astype(np.float32)

        # Lazy-init explainer
        if self._explainer is None:
            try:
                self._explainer = _shap.TreeExplainer(self._model)
            except Exception:
                try:
                    self._explainer = _shap.Explainer(self._model)
                except Exception as exc:
                    warnings.warn(f"[warn] Could not build SHAP explainer: {exc}", stacklevel=2)
                    return {}

        try:
            shap_values = self._explainer.shap_values(X)  # list of 4 arrays or 3-d array
        except Exception as exc:
            warnings.warn(f"[warn] SHAP computation failed: {exc}", stacklevel=2)
            return {}

        feature_names = self._meta["feature_names"]
        result: dict[str, list] = {}

        # shap_values may be list (one per class) or ndarray
        if isinstance(shap_values, list):
            class_shaps = shap_values
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            class_shaps = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        else:
            class_shaps = [shap_values]

        for cls_idx, cls_label in enumerate(CLASSES):
            if cls_idx >= len(class_shaps):
                break
            sv = np.asarray(class_shaps[cls_idx]).flatten()
            order = np.argsort(np.abs(sv))[::-1][:top_k]
            result[str(cls_label)] = [
                {"feature": feature_names[i], "shap_value": float(sv[i])}
                for i in order
                if i < len(feature_names)
            ]

        return result
