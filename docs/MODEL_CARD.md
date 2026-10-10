# Model Card — ED AI Triage

> **Version:** 1.0 | **Status:** Production Candidate | **Last reviewed:** see git log

---

## 1. Intended Use

**Primary purpose:** Clinical decision-support in the Emergency Department. Predicts triage acuity (1–4, matching the Emergency Severity Index) from patient demographics, structured vitals, and chief complaint text.

**Intended users:** ED triage nurses and physicians using it as a *second opinion* alongside their clinical assessment.

**Deployment context:** In-hospital ED intake workflow. The model's prediction, confidence score, and SHAP explanation are displayed to the clinician; the final triage decision always rests with a licensed clinician.

---

## 2. Out-of-Scope Use

| ❌ Not intended for | Reason |
|---|---|
| Pediatric-only EDs (< 1 yr) | Insufficient representation in training data |
| Pre-hospital / EMS triage | Different vital-collection context; not validated |
| Standalone autonomous diagnosis | Regulatory and patient-safety requirements |
| Non-ED clinical settings | Training distribution mismatch |
| Psychiatric emergencies as primary complaint | Model under-performs on this subgroup |

---

## 3. Training Data

| Property | Value |
|---|---|
| File | `master_emergency_triage_dataset.csv` |
| Row type used | ENC#### rows only (~5,000 rows) |
| Excluded | VT#### synthetic rows (no vitals collected) |
| Target | `acuity_target` ∈ {1, 2, 3, 4} |
| Class imbalance | ≈ 121 : 1 (Acuity-4 : Acuity-1) |
| Missing vitals | MICE imputation (IterativeImputer, max_iter=10, seed=42) |
| Outlier strategy | Sensor artifacts → NaN; clinical extremes preserved |

See [`docs/DATA_CARD.md`](DATA_CARD.md) for full data provenance.

---

## 4. Metrics

*Populate from `ml_runs/safety_report.json` after `make train`.*

| Metric | Threshold | Trained value |
|---|---|---|
| Macro-F1 | ≥ 0.70 | *(see safety_report.json)* |
| Weighted-F1 | ≥ 0.80 | *(see safety_report.json)* |
| Acuity-1 Recall | ≥ 0.80 | *(see safety_report.json)* |
| Under-triage Rate (overall) | ≤ 0.10 | *(see safety_report.json)* |
| **Critical-Miss Rate** | **≤ 0.02** | *(see safety_report.json)* |
| ECE (macro) | ≤ 0.08 | *(see safety_report.json)* |
| Brier Score (macro) | ≤ 0.15 | *(see safety_report.json)* |

> ⚠️ **Blocking gate:** CI fails if `critical_miss_rate > 0.02` on hold-out evaluation.

---

## 5. Subgroup Performance

Evaluated via `src/ml/fairness_audit.py`. Results written to `ml_runs/fairness_report.csv` after `make audit`.

| Protected attribute | Subgroups audited |
|---|---|
| `age_bucket` | pediatric (0–17), young_adult (18–39), middle_age (40–64), senior (65–79), elderly (80+) |
| `gender` | M, F |

A `[WARN]` is emitted if any subgroup's Acuity-1/2 sensitivity falls > 10% below the overall population sensitivity.

---

## 6. Known Limitations

- **Class imbalance ~121×** — Acuity-1 cases represent ≈ 0.4% of encounters; recall on this class requires continuous monitoring.
- **Synthetic VT rows removed** — dataset excludes records without any vitals, which may not reflect the full clinical presentation spectrum.
- **No external validation yet** — model trained and evaluated on a single-site dataset; generalisation to other EDs is unverified.
- **Chief complaint encoding** — cross-fitted target encoding captures frequency-based signal only; semantic understanding is delegated to the LLM rationale layer.
- **Single-snapshot vitals** — no time-series trending; deteriorating patients may be under-triaged if initial vitals appear stable.

---

## 7. Ethical Considerations

- **Clinician override is mandatory.** This system is decision-support only; it must never be used autonomously.
- **Subgroup audits quarterly** — disparate impact monitoring is built in; see `fairness_audit.py`.
- **PHI protection** — chief complaint is regex-scrubbed before any LLM call (`pii_scrubber.py`); audit log stores only SHA-256 hashes of encounter IDs, never raw vitals.
- **Model registry** — every deployed version is versioned and rollback-capable within 15 minutes.
- **Disclaimer on every response** — *"Decision-support only. Not a diagnosis. Confirm with a licensed clinician."*

---

## 8. Review Cadence

| Trigger | Action |
|---|---|
| **Monthly** | Drift monitoring (`check_drift`); fairness audit |
| **Quarterly** | Full re-train + re-evaluation on fresh data |
| **Yearly** | Clinical re-certification; IRB / governance review |
| PSI > 0.20 on any feature | Investigate root cause within 48 h; consider retraining |
| Critical miss rate > 2% in production | Immediate rollback + incident escalation |

See [`docs/DEPLOYMENT_RUNBOOK.md`](DEPLOYMENT_RUNBOOK.md) for step-by-step procedures.
