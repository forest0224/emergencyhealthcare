"""
ml/predict.py
=============
Production inference and SHAP explainability wrapper for ED-Triage acuity prediction.

Loads trained model artifacts (best_model.pkl, feature_meta.pkl) and applies
the identical feature engineering pipeline used during training.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.preprocessing import (
    AGE_BINS,
    AGE_LABELS,
    CLASSES,
    GENDER_MAP,
    MICE_COLS,
    clean_triage_inputs,
    transform_features,
)

# ---------------------------------------------------------------------------
# Optional: SHAP
# ---------------------------------------------------------------------------
try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    warnings.warn("[warn] shap not installed; explain() will return empty dict.", stacklevel=2)


def build_single_row(patient: dict, meta: dict) -> tuple[pd.DataFrame, list[str]]:
    """Build a one-row feature DataFrame from a patient dict matching training features."""
    d: dict[str, Any] = {k: v for k, v in patient.items()}

    # Track missing continuous vitals (to be imputed)
    imputed_fields: list[str] = [
        c for c in MICE_COLS if (c not in d or d[c] is None or (isinstance(d[c], float) and math.isnan(d[c])))
    ]

    if "outlier_bounds" in meta:
        frame = pd.DataFrame([d])
        cleaned = clean_triage_inputs(frame, meta["outlier_bounds"])
        imputed_fields = [field for field in MICE_COLS if pd.isna(cleaned[field].iat[0])]
        return transform_features(frame, meta), imputed_fields

    for c in MICE_COLS:
        if c not in d or d[c] is None:
            d[c] = np.nan

    hr   = d.get("heart_rate", np.nan)
    sbp  = d.get("systolic_bp", np.nan)
    dbp  = d.get("diastolic_bp", np.nan)
    rr   = d.get("respiratory_rate", np.nan)
    spo2 = d.get("spo2", np.nan)
    temp = d.get("temperature", np.nan)

    def _safe_div(a, b):
        try:
            if b is not None and not np.isnan(b) and b != 0:
                return float(a) / float(b)
        except Exception:
            pass
        return np.nan

    d["shock_index"]     = _safe_div(hr, sbp)
    d["map"]             = ((sbp or 0) + 2 * (dbp or 0)) / 3.0
    d["pulse_pressure"]  = (sbp or 0) - (dbp or 0)
    d["spo2_deficit"]    = 100.0 - (spo2 if spo2 is not None and not np.isnan(spo2) else np.nan)
    d["temp_deviation"]  = abs((temp if temp is not None and not np.isnan(temp) else 37.0) - 37.0)
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
    cc_val = str(d.get("chief_complaint", ""))
    global_mean = float(np.mean(list(cc_map.values()))) if cc_map else 2.5
    d["chief_complaint_enc"] = float(cc_map.get(cc_val, global_mean))

    # Gender
    gender_raw = str(d.get("gender", "")).strip()
    d["gender_enc"] = GENDER_MAP.get(gender_raw, -1)

    # Age buckets (one-hot)
    age_val = d.get("age", np.nan)
    try:
        age_val = float(age_val) if age_val is not None else np.nan
    except (ValueError, TypeError):
        age_val = np.nan
    for lbl in AGE_LABELS:
        d[lbl] = 0
    if not np.isnan(age_val):
        for i in range(len(AGE_BINS) - 1):
            if AGE_BINS[i] <= age_val <= AGE_BINS[i + 1]:
                d[AGE_LABELS[i]] = 1
                break

    feature_names: list[str] = meta["feature_names"]
    row_vals = {f: d.get(f, 0.0) for f in feature_names}
    row_df = pd.DataFrame([row_vals], columns=feature_names).fillna(0.0)

    return row_df, imputed_fields


class TriageModel:
    """Production inference wrapper for the trained ED-Triage model."""

    def __init__(self, model_dir: str = "models") -> None:
        mdir = Path(model_dir)
        model_path = mdir / "best_model.pkl"
        meta_path = mdir / "feature_meta.pkl"
        if not model_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                f"Model artifacts not found in '{mdir}'. "
                "Run `python -m ml.train` or set MODEL_DIR to a trusted artifact directory."
            )

        self._model = joblib.load(model_path)
        self._meta  = joblib.load(meta_path)
        explain_path = mdir / "explanation_model.pkl"
        self._explanation_model = joblib.load(explain_path) if explain_path.exists() else self._model
        self._model_dir = mdir
        self._label_imap: dict[int, int] = self._meta.get("label_imap", {i: i + 1 for i in range(4)})
        self._explainer: Any = None
        self.shap_output: str | None = None

    @property
    def model_name(self) -> str:
        return self._meta.get("best_model", "TrainedModel")

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    def predict_proba(self, patient_dict: dict) -> dict:
        """Run inference on a single patient dict."""
        row_df, imputed_fields = build_single_row(patient_dict, self._meta)
        X = row_df.values.astype(np.float32)

        raw_proba = self._model.predict_proba(X)

        classes_in = list(self._model.classes_)
        proba_aligned = np.zeros(4, dtype=float)
        for col_i, cls_i in enumerate(classes_in):
            orig = self._label_imap.get(int(cls_i), int(cls_i))
            if 1 <= orig <= 4:
                proba_aligned[orig - 1] = float(raw_proba[0, col_i])

        predicted = int(proba_aligned.argmax()) + 1
        confidence    = float(proba_aligned.max())

        return {
            "proba":          proba_aligned.tolist(),
            "predicted":      predicted,
            "confidence":     confidence,
            "imputed_fields": imputed_fields,
            "feature_vector": row_df.iloc[0].to_dict(),
        }

    def explain(self, patient_dict: dict, top_k: int = 5) -> dict[str, list[dict[str, Any]]]:
        """Return local SHAP explanation for a single patient."""
        if not _HAS_SHAP:
            return {}

        row_df, _ = build_single_row(patient_dict, self._meta)
        X = row_df.values.astype(np.float32)

        estimator = self._explanation_model
        if hasattr(estimator, "steps"):
            X = estimator[:-1].transform(X)
            estimator = estimator.steps[-1][1]
        if self._explainer is None:
            try:
                explainer = _shap.TreeExplainer(estimator)
                score = "Uncalibrated raw model score"
            except Exception:
                try:
                    explainer = _shap.LinearExplainer(estimator, np.zeros((1, X.shape[1])))
                    score = "Uncalibrated linear decision score (log-odds units)"
                except Exception:
                    try:
                        explainer = _shap.Explainer(
                            estimator.predict_proba, np.zeros((1, X.shape[1])),
                            algorithm="permutation", seed=42,
                        )
                        score = "Uncalibrated base-model probability"
                    except Exception as exc:
                        warnings.warn(f"[warn] Could not build SHAP explainer: {exc}", stacklevel=2)
                        return {}
            self.shap_output = score
            self._explainer = explainer

        try:
            if hasattr(self._explainer, "shap_values"):
                shap_values = self._explainer.shap_values(X)
            else:
                shap_values = self._explainer(X, max_evals=2 * X.shape[1] + 1).values

            feature_names = self._meta["feature_names"]
            classes = [self._label_imap.get(int(cls), int(cls)) for cls in estimator.classes_]
            values = _class_attributions(shap_values, len(feature_names), len(classes))
        except Exception as exc:
            warnings.warn(f"[warn] SHAP computation failed: {exc}", stacklevel=2)
            return {}

        result = {}
        for cls, contributions in zip(classes, values):
            if cls not in CLASSES:
                continue
            finite = np.flatnonzero(np.isfinite(contributions))
            ordered = finite[np.argsort(-np.abs(contributions[finite]), kind="stable")][:top_k]
            if len(ordered):
                result[str(cls)] = [
                    {"feature": feature_names[i], "shap_value": float(contributions[i])}
                    for i in ordered
                ]
        return result


def _class_attributions(values, features, classes):
    """Normalize SHAP output to (class, feature) for one observation."""
    array = np.asarray(values, dtype=float)
    if isinstance(values, list) and array.shape == (classes, 1, features):
        return array[:, 0, :]
    if array.shape == (1, features, classes):
        return array[0].T
    raise ValueError(f"Unexpected multiclass SHAP shape {array.shape}.")
