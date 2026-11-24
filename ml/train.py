"""
src/ml/tabular_models.py
=========================
Train and cross-validate multiple tabular classifiers for ED-Triage acuity
prediction.  Handles severe class imbalance inside training folds only.

Public API
----------
train_and_evaluate(X, y, *, n_splits=5, random_state=42,
                   output_dir="ml_runs") -> dict

Usage
-----
>>> from ml.preprocessing import build_features
>>> from ml.train import train_and_evaluate
>>> X, y, meta = build_features("master_emergency_triage_dataset.csv")
>>> results = train_and_evaluate(X, y)
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer, RobustScaler

from ml.preprocessing import fit_features, training_data, transform_features
from ml.calibration import calibrate
from ml.safety_metrics import compute_safety_metrics
from ml.fairness_audit import audit_fairness
from ml.drift import fit_reference

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Optional dependencies — graceful degradation
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    warnings.warn("[warn] xgboost not installed; XGBoost model skipped.", stacklevel=2)

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    warnings.warn("[warn] lightgbm not installed; LightGBM model skipped.", stacklevel=2)

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False
    warnings.warn("[warn] catboost not installed; CatBoost model skipped.", stacklevel=2)

try:
    from imblearn.over_sampling import SMOTENC, SMOTE
    _HAS_IMBLEARN = True
except ImportError:
    _HAS_IMBLEARN = False
    warnings.warn("[warn] imbalanced-learn not installed; using class_weight only.", stacklevel=2)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
CLASSES      = [1, 2, 3, 4]
_HIGH_ACUITY = {1, 2}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights.

    Args:
        y: Array of integer class labels.

    Returns:
        Dict mapping class label to weight.

    Example:
        >>> w = _class_weights(y_train)
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_cls = len(classes)
    return {int(c): total / (n_cls * cnt) for c, cnt in zip(classes, counts)}


def _make_models(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Instantiate all candidate models.

    Args:
        random_state: Seed for reproducibility.

    Returns:
        Ordered dict of {model_name: estimator}.

    Example:
        >>> models = _make_models()
    """
    models: dict[str, Any] = {}

    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            num_class=4,
            random_state=random_state,
            tree_method="hist",
            verbosity=0,
        )

    if _HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=random_state,
            verbosity=-1,
            n_jobs=-1,
        )

    if _HAS_CAT:
        models["CatBoost"] = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=0,
        )

    models["LogisticRegression"] = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        multi_class="multinomial",
        solver="lbfgs",
        random_state=random_state,
    )

    models["MLP"] = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
    )

    return models


def _smote_resample(
    X_tr: np.ndarray, y_tr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE inside a training fold if imbalanced-learn is available.

    Args:
        X_tr: Training feature array.
        y_tr: Training label array.

    Returns:
        Resampled (X, y) tuple.

    Example:
        >>> Xr, yr = _smote_resample(X_train, y_train)
    """
    if not _HAS_IMBLEARN:
        return X_tr, y_tr
    try:
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
        return sm.fit_resample(X_tr, y_tr)
    except Exception as exc:
        warnings.warn(f"[warn] SMOTE failed ({exc}); using original distribution.", stacklevel=2)
        return X_tr, y_tr


def _compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute per-fold aggregate metrics.

    Args:
        y_true:  True integer labels.
        y_pred:  Predicted integer labels.
        y_proba: Predicted probability matrix (N × 4).

    Returns:
        Dict of metric_name → float.

    Example:
        >>> m = _compute_fold_metrics(y_true, y_pred, y_proba)
    """
    report = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    macro_f1    = float(report["macro avg"]["f1-score"])
    weighted_f1 = float(report["weighted avg"]["f1-score"])
    rec1        = float(report.get("1", {}).get("recall", 0.0))
    rec2        = float(report.get("2", {}).get("recall", 0.0))

    # Under-triage: high-acuity (1,2) predicted as low-acuity (3,4)
    ha_mask   = np.isin(y_true, list(_HIGH_ACUITY))
    under_all = float(np.mean(y_pred[ha_mask] > y_true[ha_mask])) if ha_mask.sum() > 0 else 0.0

    # Macro ROC-AUC (OvR)
    lb = LabelBinarizer()
    lb.fit(CLASSES)
    y_bin = lb.transform(y_true)
    try:
        roc_auc = float(roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        roc_auc = float("nan")

    return {
        "macro_f1":          macro_f1,
        "weighted_f1":       weighted_f1,
        "acuity1_recall":    rec1,
        "acuity12_recall":   float((rec1 + rec2) / 2.0),
        "under_triage_rate": under_all,
        "roc_auc_macro":     roc_auc,
    }


def _plot_confusion(
    cm: np.ndarray, model_name: str, out_dir: Path
) -> None:
    """Save a confusion-matrix heatmap.

    Args:
        cm:         Confusion matrix array (4×4).
        model_name: Used in title and filename.
        out_dir:    Directory to save the PNG.

    Example:
        >>> _plot_confusion(cm, "XGBoost", Path("ml_runs"))
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels([str(c) for c in CLASSES])
    ax.set_yticklabels([str(c) for c in CLASSES])
    ax.set_xlabel("Predicted Acuity")
    ax.set_ylabel("True Acuity")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.colorbar(im, ax=ax)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black",
                    fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / f"confusion_{model_name}.png", dpi=120)
    plt.close(fig)


def _plot_roc_pr(
    y_true_all: np.ndarray,
    y_proba_all: np.ndarray,
    model_name: str,
    out_dir: Path,
) -> None:
    """Save ROC and PR curves for each acuity class.

    Args:
        y_true_all:  Concatenated true labels from all CV folds.
        y_proba_all: Concatenated probability arrays from all CV folds.
        model_name:  Used in title and filename.
        out_dir:     Directory to save the PNG.

    Example:
        >>> _plot_roc_pr(y_true, y_proba, "XGBoost", Path("ml_runs"))
    """
    from sklearn.metrics import roc_curve, precision_recall_curve

    lb = LabelBinarizer()
    lb.fit(CLASSES)
    y_bin = lb.transform(y_true_all)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    for idx, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_proba_all[:, idx])
        auc_val = float(np.trapz(tpr, fpr))
        axes[0].plot(fpr, tpr, color=colors[idx], label=f"Acuity {cls} (AUC={auc_val:.3f})")

        prec, rec, _ = precision_recall_curve(y_bin[:, idx], y_proba_all[:, idx])
        ap = average_precision_score(y_bin[:, idx], y_proba_all[:, idx])
        axes[1].plot(rec, prec, color=colors[idx], label=f"Acuity {cls} (AP={ap:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC Curves — {model_name}")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR Curves — {model_name}")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / f"roc_pr_{model_name}.png", dpi=120)
    plt.close(fig)


def _plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    model_name: str,
    out_dir: Path,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar chart of top feature importances.

    Args:
        importances:   Array of importance scores.
        feature_names: Corresponding feature names.
        model_name:    Used in title and filename.
        out_dir:       Directory to save the PNG.
        top_n:         Number of top features to show.

    Example:
        >>> _plot_feature_importance(imp, names, "XGBoost", Path("ml_runs"))
    """
    idx   = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(names, vals, color="#3b82d4")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name} (top {top_n})")
    plt.tight_layout()
    fig.savefig(out_dir / f"feature_importance_{model_name}.png", dpi=120)
    plt.close(fig)


def _scale_X(
    X_tr: np.ndarray, X_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit RobustScaler on X_tr, apply to X_tr and X_val.

    Args:
        X_tr:  Training feature array.
        X_val: Validation feature array.

    Returns:
        Tuple (X_tr_scaled, X_val_scaled).

    Example:
        >>> Xtr_s, Xval_s = _scale_X(X_tr, X_val)
    """
    sc = RobustScaler()
    return sc.fit_transform(X_tr), sc.transform(X_val)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    output_dir: str = "models",
    raw_data: pd.DataFrame | None = None,
) -> dict:
    """Train and stratified-CV evaluate all candidate models for ED-Triage.

    Class imbalance is handled INSIDE training folds only:
        - SMOTE oversampling (if imbalanced-learn available)
        - Else class_weight / scale_pos_weight balancing in estimators

    Writes to output_dir:
        metrics_summary.csv           - Leaderboard across models
        per_class_{model}.csv         - Per-class P/R/F1
        confusion_{model}.png         - Confusion matrix heatmap
        roc_pr_{model}.png            - ROC + PR curves
        feature_importance_{model}.png
        best_model.pkl                - Best model by macro-F1
        feature_meta.pkl              - Feature names + encoding maps

    Args:
        X:          Feature DataFrame (all numeric, no NaN).
        y:          Integer target Series (1-4).
        n_splits:   Number of stratified CV folds.
        random_state: Seed.
        output_dir: Directory for all outputs.

    Returns:
        Dict with keys: leaderboard (DataFrame), best_model_name (str),
        best_model_path (str), all_metrics (dict).

    Example:
        >>> results = train_and_evaluate(X, y, output_dir="ml_runs")
        >>> print(results["best_model_name"])
    """
    out_dir = _ensure_dir(Path(output_dir))
    feature_names = list(X.columns)
    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(int)

    # Remap labels to 0-based for XGBoost internal use; we un-remap predictions
    label_map  = {c: i for i, c in enumerate(CLASSES)}
    label_imap = {i: c for c, i in label_map.items()}
    y_mapped   = np.array([label_map[v] for v in y_arr])

    models  = _make_models(random_state)
    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_metrics: dict[str, list[dict]] = {name: [] for name in models}
    # Store oof predictions for confusion matrix + ROC/PR
    oof_preds:  dict[str, np.ndarray] = {n: np.zeros(len(y_arr), dtype=int) for n in models}
    oof_probas: dict[str, np.ndarray] = {n: np.zeros((len(y_arr), 4), dtype=float) for n in models}

    print("\n" + "=" * 70)
    print("TRAINING & CROSS-VALIDATION")
    print("=" * 70)

    prepared = []
    for tr_idx, val_idx in skf.split(X_arr, y_mapped):
        if raw_data is None:
            train_values, val_values = X_arr[tr_idx], X_arr[val_idx]
        else:
            train_features, _, fold_meta = fit_features(raw_data.iloc[tr_idx])
            val_features = transform_features(raw_data.iloc[val_idx], fold_meta)
            train_values = train_features.to_numpy(dtype=np.float32)
            val_values = val_features.to_numpy(dtype=np.float32)
        prepared.append((tr_idx, val_idx, train_values, val_values))

    for model_name, estimator in models.items():
        print(f"\n-- {model_name} --")
        for fold_i, (tr_idx, val_idx, X_tr, X_val) in enumerate(prepared):
            y_tr, y_val = y_mapped[tr_idx], y_mapped[val_idx]

            # Scale for LR and MLP
            if model_name in ("LogisticRegression", "MLP"):
                X_tr, X_val = _scale_X(X_tr, X_val)

            # SMOTE (applied before class-weight models too for uniformity)
            if model_name in ("LogisticRegression", "MLP"):
                X_tr, y_tr = _smote_resample(X_tr, y_tr)

            # XGBoost: set scale_pos_weight via sample_weight equivalent
            fit_kwargs: dict[str, Any] = {}
            if model_name == "XGBoost":
                cw = _class_weights(y_tr)
                sw = np.array([cw[v] for v in y_tr], dtype=float)
                fit_kwargs["sample_weight"] = sw

            try:
                estimator.fit(X_tr, y_tr, **fit_kwargs)
            except TypeError:
                estimator.fit(X_tr, y_tr)

            # Predict — map back to original labels
            raw_pred  = estimator.predict(X_val)
            raw_proba = estimator.predict_proba(X_val)

            # Align proba columns to CLASSES order (0→1, 1→2, 2→3, 3→4)
            pred_orig  = np.array([label_imap.get(int(p), p) for p in raw_pred.ravel()])
            classes_in = list(estimator.classes_)
            proba_aligned = np.zeros((len(val_idx), 4), dtype=float)
            for col_i, cls_i in enumerate(classes_in):
                orig_cls = label_imap.get(int(cls_i), int(cls_i))
                if orig_cls in label_map:
                    out_col = list(label_map.keys()).index(orig_cls)
                    proba_aligned[:, out_col] = raw_proba[:, col_i]

            y_val_orig = np.array([label_imap[v] for v in y_val])
            oof_preds[model_name][val_idx]   = pred_orig
            oof_probas[model_name][val_idx]  = proba_aligned

            fm = _compute_fold_metrics(y_val_orig, pred_orig, proba_aligned)
            all_metrics[model_name].append(fm)
            print(
                f"  Fold {fold_i+1}: macro_F1={fm['macro_f1']:.4f}  "
                f"Acuity-1 Recall={fm['acuity1_recall']:.4f}  "
                f"Under-triage={fm['under_triage_rate']:.4f}"
            )

    # ---------------------------------------------------------------------------
    # Aggregate metrics + write per-model artefacts
    # ---------------------------------------------------------------------------
    summary_rows: list[dict] = []
    best_name: str = ""
    best_f1:   float = -1.0
    best_model_path = str(out_dir / "best_model.pkl")

    for model_name, estimator in models.items():
        fold_metrics = all_metrics[model_name]
        agg = {k: float(np.mean([f[k] for f in fold_metrics])) for k in fold_metrics[0]}

        flag = "⚠ " if agg["acuity1_recall"] < 0.80 else "  "
        summary_rows.append({"model": model_name, **agg, "flag": flag})

        # Confusion matrix (OOF)
        cm = confusion_matrix(y_arr, oof_preds[model_name], labels=CLASSES)
        _plot_confusion(cm, model_name, out_dir)

        # ROC / PR
        _plot_roc_pr(y_arr, oof_probas[model_name], model_name, out_dir)

        # Per-class CSV
        report = classification_report(
            y_arr, oof_preds[model_name],
            labels=CLASSES, output_dict=True, zero_division=0,
        )
        per_class_rows = [
            {"class": cls, **report[str(cls)]}
            for cls in CLASSES if str(cls) in report
        ]
        pd.DataFrame(per_class_rows).to_csv(
            out_dir / f"per_class_{model_name}.csv", index=False
        )

        # Feature importance
        if hasattr(estimator, "feature_importances_"):
            _plot_feature_importance(
                estimator.feature_importances_, feature_names,
                model_name, out_dir,
            )
        elif hasattr(estimator, "coef_"):
            imp = np.abs(estimator.coef_).mean(axis=0)
            _plot_feature_importance(imp, feature_names, model_name, out_dir)

        # Track best
        if agg["macro_f1"] > best_f1:
            best_f1   = agg["macro_f1"]
            best_name = model_name

    # ---------------------------------------------------------------------------
    # Retrain best model on full data, save artefacts
    # ---------------------------------------------------------------------------
    best_estimator = models[best_name]
    X_full = X_arr.copy()
    y_full = y_mapped.copy()
    if best_name in ("LogisticRegression", "MLP"):
        sc = RobustScaler()
        X_full = sc.fit_transform(X_full)
        X_full, y_full = _smote_resample(X_full, y_full)
    fit_kwargs_full: dict[str, Any] = {}
    if best_name == "XGBoost":
        cw_full = _class_weights(y_full)
        sw_full = np.array([cw_full[v] for v in y_full], dtype=float)
        fit_kwargs_full["sample_weight"] = sw_full
    try:
        best_estimator.fit(X_full, y_full, **fit_kwargs_full)
    except TypeError:
        best_estimator.fit(X_full, y_full)

    if best_name in ("LogisticRegression", "MLP"):
        best_estimator = make_pipeline(sc, best_estimator)
    joblib.dump(best_estimator, best_model_path)
    feature_meta = {
        "feature_names": feature_names,
        "label_map":     label_map,
        "label_imap":    label_imap,
        "best_model":    best_name,
    }
    joblib.dump(feature_meta, out_dir / "feature_meta.pkl")
    print(f"\n[saved] best_model.pkl ({best_name}) → {best_model_path}")

    # ---------------------------------------------------------------------------
    # Leaderboard
    # ---------------------------------------------------------------------------
    leaderboard = (
        pd.DataFrame(summary_rows)
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )
    leaderboard.to_csv(out_dir / "metrics_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("LEADERBOARD (sorted by Macro-F1)")
    print("=" * 70)
    cols_show = ["flag", "model", "macro_f1", "weighted_f1",
                 "acuity1_recall", "acuity12_recall",
                 "under_triage_rate", "roc_auc_macro"]
    print(leaderboard[cols_show].to_string(index=False))
    print("\n⚠  = Acuity-1 Recall < 0.80  (clinical safety threshold)")

    return {
        "leaderboard":      leaderboard,
        "best_model_name":  best_name,
        "best_model_path":  best_model_path,
        "all_metrics":      all_metrics,
    }


# ---------------------------------------------------------------------------


def main() -> None:
    """Train, calibrate, and evaluate the selected model on independent splits."""
    import argparse
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--dataset", default="data/dataset.csv")
    parser.add_argument("--output-dir", default=os.getenv("MODEL_DIR", "models"))
    args = parser.parse_args()
    data = training_data(args.dataset)
    training, held_out = train_test_split(
        data, test_size=0.30, stratify=data.acuity_target, random_state=RANDOM_STATE
    )
    calibration, evaluation = train_test_split(
        held_out, test_size=0.50, stratify=held_out.acuity_target, random_state=RANDOM_STATE
    )
    X, y, meta = fit_features(training)
    out = Path(args.output_dir)
    result = train_and_evaluate(X, y, output_dir=str(out), raw_data=training)
    base = joblib.load(out / "best_model.pkl")
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    X_cal = transform_features(calibration, meta).to_numpy(dtype=np.float32)
    X_eval = transform_features(evaluation, meta).to_numpy(dtype=np.float32)
    model = calibrate(base, X_cal, calibration.acuity_target.to_numpy() - 1, method="sigmoid")
    probabilities = model.predict_proba(X_eval)
    predictions = probabilities.argmax(axis=1) + 1
    truth = evaluation.acuity_target.to_numpy()
    safety = compute_safety_metrics(truth, predictions, probabilities)
    groups = evaluation.copy()
    groups["age_bucket"] = pd.cut(groups.age, [-1, 17, 39, 64, 79, 120])
    fairness = audit_fairness(groups.reset_index(drop=True), truth, predictions)
    report = {
        "leaderboard": json.loads(result["leaderboard"].to_json(orient="records")),
        "held_out": {
            **_compute_fold_metrics(truth, predictions, probabilities),
            "accuracy": accuracy_score(truth, predictions),
            "macro_precision": precision_score(truth, predictions, average="macro", zero_division=0),
            "macro_recall": recall_score(truth, predictions, average="macro", zero_division=0),
        },
        "confusion_matrix": confusion_matrix(truth, predictions, labels=CLASSES).tolist(),
        "safety": safety,
        "fairness": json.loads(fairness.to_json(orient="records")),
        "calibrated": model is not base,
        "evaluated_rows": len(evaluation),
    }
    report = json.loads(pd.Series(report).to_json())
    meta.update(joblib.load(out / "feature_meta.pkl"))
    meta.update(version=version, dataset_rows=len(data), evaluation=report)
    joblib.dump(base, out / "explanation_model.pkl")
    joblib.dump(model, out / "best_model.pkl")
    joblib.dump(meta, out / "feature_meta.pkl")
    fit_reference(X, ref_path=str(out / "drift_reference.pkl"))
    (out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
