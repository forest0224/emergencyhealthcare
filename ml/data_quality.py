"""Dataset diagnostics and duplicate-aware evaluation partitions."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ml.preprocessing import CLASSES, INPUT_BOUNDS, MICE_COLS, TARGET_COL

MODEL_INPUTS = [*MICE_COLS, "arrival_hour", "gender"]


def evaluation_groups(frame: pd.DataFrame) -> np.ndarray:
    """Join rows sharing a patient, encounter, or canonical model input."""
    rows = frame.reset_index(drop=True)
    features = rows[MODEL_INPUTS].copy()
    for name, (low, high) in INPUT_BOUNDS.items():
        features[name] = features[name].where(features[name].between(low, high))
    inverted = features.systolic_bp <= features.diastolic_bp
    features.loc[inverted, ["systolic_bp", "diastolic_bp"]] = np.nan
    features["gender"] = features["gender"].fillna("").str.strip().str.lower().replace(
        {"male": "m", "female": "f"}
    )
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    keys = [pd.util.hash_pandas_object(features, index=False)]
    for name in ("patient_id", "subject_id", "encounter_id"):
        if name in rows:
            keys.append(rows[name].astype("string").str.strip().replace("", pd.NA))
    for values in keys:
        seen = {}
        for i, value in enumerate(values):
            if pd.notna(value):
                other = seen.setdefault(value, i)
                parent[root(i)] = root(other)
    return np.array([root(i) for i in range(len(rows))])


def split_folds(labels, groups, count=5, seed=42):
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(labels)), labels, groups))
    for training, validation in folds:
        if set(labels[training]) != set(CLASSES) or set(labels[validation]) != set(CLASSES):
            raise ValueError("Every partition requires all four labels in independent groups.")
    return folds


def holdout(labels, groups, seed=42):
    """Choose the stratified fold nearest 20% of each class and total rows."""
    labels = np.asarray(labels)
    target = np.array([np.sum(labels == cls) / 5 for cls in CLASSES])

    def error(fold):
        _, indices = fold
        counts = np.array([np.sum(labels[indices] == cls) for cls in CLASSES])
        return float(np.mean(abs(counts - target) / target))

    return min(split_folds(labels, groups, seed=seed), key=error)


def dataset_audit(path, eligible):
    raw = pd.read_csv(path)
    numeric = eligible.select_dtypes(include="number")
    complaint = eligible.chief_complaint.fillna("").str.strip().str.lower()
    by_complaint = eligible.groupby(complaint)[TARGET_COL].agg(["size", "nunique"])
    inputs = eligible.drop(columns=["encounter_id", TARGET_COL], errors="ignore").copy()
    inputs["chief_complaint"] = complaint
    lookup_scores = []
    labels = eligible[TARGET_COL]
    for training, validation in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(complaint, labels):
        lookup = labels.iloc[training].groupby(complaint.iloc[training]).agg(lambda group: group.mode().iloc[0])
        guesses = complaint.iloc[validation].map(lookup).fillna(labels.iloc[training].mode().iloc[0])
        lookup_scores.append(f1_score(labels.iloc[validation], guesses, average="macro"))
    return {
        "diagnostic_complaint_lookup_macro_f1": float(np.mean(lookup_scores)),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "raw_rows": len(raw), "eligible_rows": len(eligible),
        "types": raw.dtypes.astype(str).to_dict(), "missing": raw.isna().sum().to_dict(),
        "exact_duplicates": int(raw.duplicated().sum()),
        "repeated_encounters": int(raw.encounter_id.duplicated().sum()),
        "duplicate_inputs": int(inputs.duplicated().sum()),
        "conflicting_input_groups": int(eligible.groupby(
            pd.util.hash_pandas_object(inputs, index=False)
        )[TARGET_COL].nunique().gt(1).sum()),
        "class_counts": eligible[TARGET_COL].astype(int).value_counts().sort_index().to_dict(),
        "complaint_categories": len(by_complaint),
        "single_label_complaint_rows": int(by_complaint.loc[by_complaint["nunique"].eq(1), "size"].sum()),
        "out_of_bounds": {col: int((eligible[col].notna() & ~eligible[col].between(*bounds)).sum())
                          for col, bounds in INPUT_BOUNDS.items()},
        "inverted_bp": int((eligible.systolic_bp <= eligible.diastolic_bp).sum()),
        "correlation": numeric.corr().round(5).to_dict(),
        "class_quantiles": [
            {"acuity": cls, "quantile": q, **values}
            for (cls, q), values in eligible.groupby(TARGET_COL)[list(INPUT_BOUNDS)]
                .quantile([0, .05, .5, .95, 1]).to_dict(orient="index").items()
        ],
    }
