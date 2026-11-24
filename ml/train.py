"""
ml/train.py
=========================
Train and cross-validate multiple tabular classifiers for ED-Triage acuity
prediction.  Handles severe class imbalance inside training folds only.

Public API
----------
train_and_evaluate(X, y, *, n_splits=5, random_state=42,
                   output_dir="models") -> dict

Usage
-----
>>> from ml.preprocessing import build_features
>>> from ml.train import train_and_evaluate
>>> X, y, meta = build_features("data/dataset.csv")
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
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer, RobustScaler

from ml.preprocessing import INPUT_BOUNDS, fit_features, training_data, transform_features
from ml.data_quality import dataset_audit, evaluation_groups, holdout, split_folds
from ml.simulate import simulate
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
            n_jobs=4,
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
            n_jobs=4,
        )

    if _HAS_CAT:
        models["CatBoost"] = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=0,
            thread_count=4,
            allow_writing_files=False,
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


REGULARIZATION_GRID = {
    "XGBoost": {"max_depth": 3, "min_child_weight": 3},
    "LightGBM": {"max_depth": 3, "num_leaves": 8, "min_child_samples": 30},
    "CatBoost": {"depth": 4, "l2_leaf_reg": 6},
    "LogisticRegression": {"C": .2},
    "MLP": {"alpha": .01},
}


def _configured_model(name, parameters):
    return _make_models()[name].set_params(**parameters)


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
        >>> _plot_confusion(cm, "XGBoost", Path("models"))
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
        >>> _plot_roc_pr(y_true, y_proba, "XGBoost", Path("models"))
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
        >>> _plot_feature_importance(imp, names, "XGBoost", Path("models"))
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
    group_ids: np.ndarray | None = None,
    fit_final: bool = True,
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
        >>> results = train_and_evaluate(X, y, output_dir="models")
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

    defaults = _make_models(random_state)
    configurations = {name: {} for name in defaults}
    if raw_data is not None:
        configurations.update({f"{name} (regularized)": params
                               for name, params in REGULARIZATION_GRID.items() if name in defaults})
    models = {
        key: clone(defaults[key.split(" (")[0]]).set_params(**params)
        for key, params in configurations.items()
    }
    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_metrics: dict[str, list[dict]] = {name: [] for name in models}
    # Store oof predictions for confusion matrix + ROC/PR
    oof_preds:  dict[str, np.ndarray] = {n: np.zeros(len(y_arr), dtype=int) for n in models}
    oof_probas: dict[str, np.ndarray] = {n: np.zeros((len(y_arr), 4), dtype=float) for n in models}

    print("\n" + "=" * 70)
    print("TRAINING & CROSS-VALIDATION")
    print("=" * 70)

    prepared = []
    folds = (split_folds(y_arr, group_ids, n_splits, random_state)
             if group_ids is not None else skf.split(X_arr, y_mapped))
    baseline_scores = []
    for tr_idx, val_idx in folds:
        if raw_data is None:
            train_values, val_values = X_arr[tr_idx], X_arr[val_idx]
        else:
            train_features, _, fold_meta = fit_features(raw_data.iloc[tr_idx])
            feature_names = fold_meta['feature_names']
            val_features = transform_features(raw_data.iloc[val_idx], fold_meta)
            train_values = train_features.to_numpy(dtype=np.float32)
            val_values = val_features.to_numpy(dtype=np.float32)
        prepared.append((tr_idx, val_idx, train_values, val_values))
        dummy = DummyClassifier(strategy="most_frequent").fit(train_values, y_arr[tr_idx])
        baseline_scores.append(f1_score(
            y_arr[val_idx], dummy.predict(val_values), labels=CLASSES, average="macro", zero_division=0
        ))

    for variant, estimator in models.items():
        model_name = variant.split(" (")[0]
        print(f"\n-- {variant} --")
        for fold_i, (tr_idx, val_idx, X_tr, X_val) in enumerate(prepared):
            y_tr, y_val = y_mapped[tr_idx], y_mapped[val_idx]

            # Scale for LR and MLP
            if model_name in ("LogisticRegression", "MLP"):
                X_tr, X_val = _scale_X(X_tr, X_val)

            observed_train = X_tr
            # LR already balances classes through weights.
            if model_name == "MLP":
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
            oof_preds[variant][val_idx]   = pred_orig
            oof_probas[variant][val_idx]  = proba_aligned

            fm = _compute_fold_metrics(y_val_orig, pred_orig, proba_aligned)
            train_labels = estimator.predict(observed_train).ravel() + 1
            fm.update(
                training_accuracy=float(accuracy_score(y_arr[tr_idx], train_labels)),
                training_macro_f1=float(f1_score(
                    y_arr[tr_idx], train_labels, labels=CLASSES, average="macro", zero_division=0
                )),
                validation_accuracy=float(accuracy_score(y_val_orig, pred_orig)),
                critical_miss_rate=float((pred_orig[y_val_orig == 1] >= 3).mean())
                    if (y_val_orig == 1).any() else 0.0,
            )
            all_metrics[variant].append(fm)
            print(
                f"  Fold {fold_i+1}: macro_F1={fm['macro_f1']:.4f}  "
                f"Acuity-1 Recall={fm['acuity1_recall']:.4f}  "
                f"Under-triage={fm['under_triage_rate']:.4f}"
            )

    configuration_scores = {
        variant: float(np.mean([fold["macro_f1"] for fold in scores]))
        for variant, scores in all_metrics.items()
    }
    winners = {
        name: max((variant for variant in models if variant.split(" (")[0] == name),
                  key=configuration_scores.__getitem__)
        for name in defaults
    }
    selected_parameters = {name: configurations[variant] for name, variant in winners.items()}
    models = {name: models[variant] for name, variant in winners.items()}
    all_metrics = {name: all_metrics[variant] for name, variant in winners.items()}
    oof_preds = {name: oof_preds[variant] for name, variant in winners.items()}
    oof_probas = {name: oof_probas[variant] for name, variant in winners.items()}

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

        agg["macro_f1_std"] = float(np.std([f["macro_f1"] for f in fold_metrics], ddof=1))
        agg["training_validation_gap"] = agg["training_macro_f1"] - agg["macro_f1"]
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
    if fit_final:
        best_estimator = models[best_name]
        X_full = X_arr.copy()
        y_full = y_mapped.copy()
        if best_name in ("LogisticRegression", "MLP"):
            sc = RobustScaler()
            X_full = sc.fit_transform(X_full)
            if best_name == "MLP":
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
        "baseline_macro_f1": float(np.mean(baseline_scores)),
        "selected_parameters": selected_parameters,
        "configuration_scores": configuration_scores,
    }


# ---------------------------------------------------------------------------


def _evaluate(model, frame, meta):
    matrix = transform_features(frame, meta).to_numpy(dtype=np.float32)
    probabilities = model.predict_proba(matrix)
    truth = frame.acuity_target.to_numpy(dtype=int)
    predictions = probabilities.argmax(axis=1) + 1
    return truth, predictions, probabilities


def _stress_tests(model, evaluation, meta, clean_macro_f1):
    rng = np.random.default_rng(RANDOM_STATE)
    measure = list(INPUT_BOUNDS)[2:]
    scenarios = {}
    missing = evaluation.copy()
    missing[measure] = missing[measure].mask(rng.random((len(missing), len(measure))) < .30)
    scenarios["30% random measurement masking"] = missing
    jitter = evaluation.copy()
    for col, sd in zip(measure, [5, 4, 4, 1.5, .2, 1, .5]):
        jitter[col] = (jitter[col] + rng.normal(0, sd, len(jitter))).clip(*INPUT_BOUNDS[col])
    both_bp = jitter.systolic_bp.notna() & jitter.diastolic_bp.notna()
    jitter.loc[both_bp, "diastolic_bp"] = np.minimum(
        jitter.loc[both_bp, "diastolic_bp"], jitter.loc[both_bp, "systolic_bp"] - 1
    )
    scenarios["Measurement variation"] = jitter
    unusual = evaluation.copy()
    unusual["heart_rate"] = 55
    unusual["spo2"] = 89
    scenarios["Contrasting pulse and oxygen readings"] = unusual
    scenarios["Shifted simulator with overlapping complaints"] = simulate(2000, seed=171, shift=True)
    rows = []
    for name, frame in scenarios.items():
        truth, prediction, probability = _evaluate(model, frame, meta)
        scores = compute_safety_metrics(truth, prediction, probability)
        rows.append({"scenario": name, "rows": len(frame), "macro_f1": scores["macro_f1"],
                     "critical_miss_rate": scores["critical_miss_rate"],
                     "macro_f1_change": scores["macro_f1"] - clean_macro_f1
                         if frame is not scenarios["Shifted simulator with overlapping complaints"] else None})
    return rows


def _calibration_points(truth, probability, name):
    rows = []
    for i, cls in enumerate(CLASSES):
        observed, predicted = calibration_curve(truth == cls, probability[:, i], n_bins=10)
        rows.extend({"class": cls, "probability": p, "observed": o, "model": name}
                    for p, o in zip(predicted, observed))
    return rows


def _learning_curve(training, groups, name, parameters, full_metrics):
    """Single fixed development fold with two group-disjoint smaller refits."""
    tr, va = split_folds(training.acuity_target, groups)[0]
    validation = training.iloc[va]
    reservoir = training.iloc[tr]
    partitions = split_folds(reservoir.acuity_target, groups[tr])
    rows = []
    for count in (1, 3):
        indices = np.concatenate([fold[1] for fold in partitions[:count]])
        subset = reservoir.iloc[indices]
        X, labels, meta = fit_features(subset)
        matrix = X.to_numpy(dtype=np.float32)
        y = labels.to_numpy() - 1
        model = _configured_model(name, parameters)
        if name in ("MLP", "LogisticRegression"):
            scaler = RobustScaler().fit(matrix)
            matrix = scaler.transform(matrix)
        if name == "MLP":
            matrix, y = _smote_resample(matrix, y)
        kwargs = {}
        if name == "XGBoost":
            weights = _class_weights(y)
            kwargs["sample_weight"] = np.array([weights[value] for value in y])
        model.fit(matrix, y, **kwargs)
        if name in ("MLP", "LogisticRegression"):
            model = make_pipeline(scaler, model)
        truth, predicted, _ = _evaluate(model, validation, meta)
        rows.append({
            "training_rows": len(subset),
            "training_macro_f1": f1_score(labels, model.predict(X.to_numpy(dtype=np.float32)).ravel() + 1,
                                           labels=CLASSES, average="macro", zero_division=0),
            "validation_macro_f1": f1_score(truth, predicted, labels=CLASSES, average="macro", zero_division=0),
        })
    rows.append({"training_rows": len(reservoir),
                 "training_macro_f1": full_metrics["training_macro_f1"],
                 "validation_macro_f1": full_metrics["macro_f1"]})
    return rows


def main() -> None:
    """Evaluate the selected classifier once on a group-disjoint test set."""
    import argparse
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--dataset", default="data/dataset.csv")
    parser.add_argument("--source", default="Original project CSV (provenance unverified)")
    parser.add_argument("--output-dir", default=os.getenv("MODEL_DIR", "models"))
    args = parser.parse_args()
    data = training_data(args.dataset)
    audit = dataset_audit(args.dataset, data)
    group_ids = evaluation_groups(data)
    development_idx, test_idx = holdout(data.acuity_target, group_ids)
    development = data.iloc[development_idx].reset_index(drop=True)
    development_groups = group_ids[development_idx]
    evaluation = data.iloc[test_idx].reset_index(drop=True)

    y = development.acuity_target.astype(int)
    X = pd.DataFrame(index=development.index)
    out = _ensure_dir(Path(args.output_dir))
    result = train_and_evaluate(X, y, output_dir=str(out), raw_data=development,
                                group_ids=development_groups, fit_final=False)
    selected = result["best_model_name"]
    selected_parameters = result["selected_parameters"][selected]
    learning_curve = _learning_curve(
        development, development_groups, selected, selected_parameters, result["all_metrics"][selected][0]
    )

    fit_idx, calibration_idx = holdout(development.acuity_target, development_groups, seed=43)
    training, calibration = development.iloc[fit_idx], development.iloc[calibration_idx]
    X, y, meta = fit_features(training)
    matrix = X.to_numpy(dtype=np.float32)
    labels = y.to_numpy() - 1
    base = _configured_model(selected, selected_parameters)
    if selected in ("LogisticRegression", "MLP"):
        scaler = RobustScaler().fit(matrix)
        matrix = scaler.transform(matrix)
    if selected == "MLP":
        matrix, labels = _smote_resample(matrix, labels)
    fit_kwargs = {}
    if selected == "XGBoost":
        weights = _class_weights(labels)
        fit_kwargs["sample_weight"] = np.array([weights[value] for value in labels])
    base.fit(matrix, labels, **fit_kwargs)
    if selected in ("LogisticRegression", "MLP"):
        base = make_pipeline(scaler, base)
    x_cal = transform_features(calibration, meta).to_numpy(dtype=np.float32)
    model = calibrate(base, x_cal, calibration.acuity_target.to_numpy() - 1,
                      method="sigmoid", always_fit=True)
    truth, predictions, probabilities = _evaluate(model, evaluation, meta)
    _, base_prediction, base_probability = _evaluate(base, evaluation, meta)
    safety = compute_safety_metrics(truth, predictions, probabilities)
    base_safety = compute_safety_metrics(truth, base_prediction, base_probability)
    subgroup = evaluation.copy()
    subgroup["age_bucket"] = pd.cut(subgroup.age, [-1, 17, 39, 64, 79, 120])
    fairness = audit_fairness(subgroup, truth, predictions)
    by_class = classification_report(truth, predictions, labels=CLASSES, output_dict=True, zero_division=0)

    report = {
        "protocol": "Five-fold stratified group CV; ~80% development, ~20% independent holdout. "
                    "Hyperparameters and models ranked by development macro F1. Sigmoid fit on reserved development data.",
        "dataset_source": args.source, "selected_model": selected,
        "dataset_audit": audit, "development_rows": len(development),
        "fit_rows": len(training), "calibration_rows": len(calibration), "evaluated_rows": len(evaluation),
        "configuration_scores": result["configuration_scores"],
        "selected_parameters": result["selected_parameters"],
        "cv_folds": result["all_metrics"], "baseline_macro_f1": result["baseline_macro_f1"],
        "leaderboard": json.loads(result["leaderboard"].to_json(orient="records")),
        "held_out": {
            **_compute_fold_metrics(truth, predictions, probabilities),
            "accuracy": accuracy_score(truth, predictions),
            "balanced_accuracy": balanced_accuracy_score(truth, predictions),
            "macro_precision": precision_score(truth, predictions, average="macro", zero_division=0),
            "macro_recall": recall_score(truth, predictions, average="macro", zero_division=0),
        },
        "per_class": [{"class": cls, **by_class[str(cls)]} for cls in CLASSES],
        "confusion_matrix": confusion_matrix(truth, predictions, labels=CLASSES).tolist(),
        "safety": safety, "fairness": json.loads(fairness.to_json(orient="records")),
        "calibrated": model is not base,
        "calibration": {
            "method": "Sigmoid; one-vs-rest macro ECE with ten equal-width bins.",
            "uncalibrated": {"log_loss": log_loss(truth, base_probability, labels=CLASSES),
                             **{key: base_safety[key] for key in ("ece_macro", "brier_macro")}},
            "deployed": {"log_loss": log_loss(truth, probabilities, labels=CLASSES),
                         **{key: safety[key] for key in ("ece_macro", "brier_macro")}},
            "points": _calibration_points(truth, base_probability, "Base")
                      + _calibration_points(truth, probabilities, "Deployed"),
        },
        "learning_curve": learning_curve,
        "robustness": _stress_tests(model, evaluation, meta, safety["macro_f1"]),
    }
    report = json.loads(pd.Series(report).to_json())
    meta.update(best_model=selected, label_imap=dict(enumerate(CLASSES)))
    meta.update(version=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
                dataset_rows=len(data), evaluation=report)
    joblib.dump(base, out / "explanation_model.pkl")
    joblib.dump(model, out / "best_model.pkl")
    joblib.dump(meta, out / "feature_meta.pkl")
    fit_reference(X, ref_path=str(out / "drift_reference.pkl"))
    (out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
