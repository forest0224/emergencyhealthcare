"""
run_full_pipeline.py
=====================
CLI entry point — imports and calls src/ml/ and src/api/ logic only.

Usage:
  python run_full_pipeline.py --mode train
  python run_full_pipeline.py --mode infer --patient demo_patient.json
  python run_full_pipeline.py --mode demo
  python run_full_pipeline.py --mode serve
  python run_full_pipeline.py --mode test
  python run_full_pipeline.py --mode audit
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# File logging
# ---------------------------------------------------------------------------
Path("logs").mkdir(exist_ok=True)
_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = f"logs/run_{_ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CSV_PATH  = "master_emergency_triage_dataset.csv"
MODEL_DIR = os.environ.get("MODEL_DIR", "ml_runs")

_CANNED = [
    {"age": 65, "gender": "M", "chief_complaint": "chest pain",
     "arrival_hour": 14, "systolic_bp": 80, "heart_rate": 122,
     "respiratory_rate": 24, "temperature": 37.2, "spo2": 93, "pain_score": 8},
    {"age": 35, "gender": "F", "chief_complaint": "shortness of breath",
     "arrival_hour": 3,  "systolic_bp": 105, "heart_rate": 98,
     "respiratory_rate": 20, "temperature": 38.5, "spo2": 97, "pain_score": 4},
    {"age": 22, "gender": "M", "chief_complaint": "ankle sprain",
     "arrival_hour": 18, "systolic_bp": 120, "heart_rate": 72,
     "respiratory_rate": 14, "temperature": 36.8, "spo2": 99, "pain_score": 3},
]

_DISCLAIMER = (
    "\n⚠  Decision-support only. Not a diagnosis. "
    "Confirm with a licensed clinician."
)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_train() -> None:
    log.info("MODE: train")
    from src.ml.feature_engineering import build_features
    from src.ml.tabular_models import train_and_evaluate
    from src.ml.safety_metrics import compute_safety_metrics
    import numpy as np

    X, y, _ = build_features(CSV_PATH)
    result   = train_and_evaluate(X, y, output_dir=MODEL_DIR)

    # Safety metrics on OOF predictions (best model)
    best_name  = result["best_model_name"]
    fold_preds = result["all_metrics"][best_name]
    # Flatten fold predictions for a representative safety snapshot
    y_arr = y.values
    import joblib, numpy as np
    model = joblib.load(f"{MODEL_DIR}/best_model.pkl")
    meta  = joblib.load(f"{MODEL_DIR}/feature_meta.pkl")
    lmap  = meta.get("label_imap", {i: i + 1 for i in range(4)})
    raw   = model.predict(X.values.astype("float32"))
    y_pred = np.array([lmap.get(int(p), p) for p in raw])
    safety = compute_safety_metrics(y_arr, y_pred)

    report_path = Path(MODEL_DIR) / "safety_report.json"
    with open(report_path, "w") as fh:
        json.dump(safety, fh, indent=2, default=str)
    log.info("Safety report -> %s", report_path)
    log.info("Critical miss rate: %.4f", safety.get("critical_miss_rate", float("nan")))
    log.info("Log -> %s", _log_file)


def mode_infer(patient_path: str) -> None:
    log.info("MODE: infer  (%s)", patient_path)
    from src.ml.inference_api import TriageModel
    from src.api.fallback_rationale import rule_based_rationale

    with open(patient_path) as fh:
        patient = json.load(fh)

    tm  = TriageModel(model_dir=MODEL_DIR)
    out = tm.predict_proba(patient)

    log.info("Predicted acuity: %d  confidence: %.1f%%", out["predicted"], out["confidence"] * 100)
    if out["imputed_fields"]:
        log.info("Imputed: %s", out["imputed_fields"])

    fb = rule_based_rationale(patient, out)
    for action in fb["immediate_actions"]:
        log.info("  • %s", action)
    print(_DISCLAIMER)


def mode_demo() -> None:
    log.info("MODE: demo")
    # Auto-train if ml_runs is empty
    if not (Path(MODEL_DIR) / "best_model.pkl").exists():
        log.info("No trained model found — running train first.")
        mode_train()

    from src.ml.inference_api import TriageModel
    from src.api.fallback_rationale import rule_based_rationale

    tm = TriageModel(model_dir=MODEL_DIR)
    for i, patient in enumerate(_CANNED, 1):
        log.info("--- Demo patient %d ---", i)
        out = tm.predict_proba(patient)
        fb  = rule_based_rationale(patient, out)
        log.info("  Acuity %d  (%.0f%%)  %s",
                 out["predicted"], out["confidence"] * 100,
                 patient["chief_complaint"])
        for action in fb["immediate_actions"][:3]:
            log.info("    • %s", action)
    print(_DISCLAIMER)


def mode_serve() -> None:
    log.info("MODE: serve")
    os.execvp("uvicorn", [
        "uvicorn", "src.api.main:app",
        "--host", "0.0.0.0", "--port", "8000", "--reload",
    ])


def mode_test() -> None:
    log.info("MODE: test")
    sys.exit(subprocess.run([sys.executable, "-m", "pytest", "-q"], check=False).returncode)


def mode_audit() -> None:
    log.info("MODE: audit")
    from src.ml.feature_engineering import build_features
    from src.ml.fairness_audit import audit_fairness
    from src.ml.drift_monitor import check_drift, fit_reference
    from src.ml.inference_api import TriageModel
    import numpy as np

    X, y, meta = build_features(CSV_PATH)
    tm = TriageModel(model_dir=MODEL_DIR)
    y_pred = np.array([tm.predict_proba(r.to_dict())["predicted"] for _, r in X.iterrows()])

    X_audit = X.copy()
    if meta.get("age_buckets") is not None:
        X_audit["age_bucket"] = meta["age_buckets"]
    if meta.get("gender") is not None:
        X_audit["gender"] = meta["gender"]

    report = audit_fairness(X_audit, y, y_pred)
    log.info("\nFairness report:\n%s", report.to_string())
    report.to_csv(f"{MODEL_DIR}/fairness_report.csv", index=False)

    ref = f"{MODEL_DIR}/drift_ref.pkl"
    if Path(ref).exists():
        drift = check_drift(X, ref_path=ref)
        log.info("Drift (top 5):\n%s", drift.head(5).to_string())
    else:
        log.warning("No drift_ref.pkl found — run train first.")
    log.info("Log -> %s", _log_file)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="ED-Triage Pipeline CLI")
    ap.add_argument("--mode", required=True,
                    choices=["train", "infer", "demo", "serve", "test", "audit"])
    ap.add_argument("--patient", default=None,
                    help="JSON file path (required for --mode infer)")
    args = ap.parse_args()

    dispatch = {
        "train": mode_train,
        "demo":  mode_demo,
        "serve": mode_serve,
        "test":  mode_test,
        "audit": mode_audit,
    }

    if args.mode == "infer":
        if not args.patient:
            ap.error("--patient <path> required for --mode infer")
        mode_infer(args.patient)
    else:
        dispatch[args.mode]()


if __name__ == "__main__":
    main()

