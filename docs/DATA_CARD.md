# Data Card — ED-Triage Dataset

> **File:** `master_emergency_triage_dataset.csv` | **Version:** 1.0

---

## Source

Single-site ED registry *(placeholder — replace with your site name and system)*.

---

## Collection Window

`<fill in — e.g. Jan 2019 – Dec 2023>`

---

## Geographic Scope

`<fill in — e.g. Urban tertiary-care hospital, North America>`

---

## Schema

| Column | Type | Range / Values | Description |
|---|---|---|---|
| `encounter_id` | string | ENC##### | Unique encounter ID. VT#### rows excluded. |
| `age` | float | 0–120 | Patient age in years |
| `gender` | string | M, F | Biological sex |
| `chief_complaint` | string | free text | Primary presenting complaint (PII-scrubbed before ML) |
| `arrival_hour` | int | 0–23 | Hour of ED arrival (24 h clock) |
| `systolic_bp` | float | 40–300 mmHg | Systolic blood pressure |
| `diastolic_bp` | float | 20–200 mmHg | Diastolic blood pressure |
| `heart_rate` | float | 20–300 bpm | Heart rate |
| `respiratory_rate` | float | 4–60 /min | Respiratory rate |
| `temperature` | float | 30–43 °C | Body temperature |
| `spo2` | float | 50–100 % | Peripheral oxygen saturation |
| `pain_score` | float | 0–10 | Self-reported pain scale |
| `acuity_target` | int | 1, 2, 3, 4 | Assigned ESI acuity level (ground truth) |

---

## Inclusion / Exclusion

**Included:**
- ENC#### encounters with at least one vital recorded and a valid `acuity_target` ∈ {1, 2, 3, 4}.

**Excluded:**
- VT#### synthetic rows (no vitals collected).
- Rows where `acuity_target` is null or outside {1, 2, 3, 4}.

---

## Target Distribution

| Acuity | Meaning | Approx. prevalence |
|---|---|---|
| 1 | **Critical** — immediate life-threatening | ≈ 0.4% |
| 2 | **Emergent** — high-risk, unstable | ≈ 4–6% |
| 3 | **Urgent** — stable, requires workup | ≈ 35–40% |
| 4 | **Non-Urgent** — minor complaint | ≈ 55–60% |

**Imbalance ratio (max / min):** ≈ 121.

---

## Known Biases

| Bias | Description | Mitigation |
|---|---|---|
| **Single-site** | Data from one hospital; may not generalise | Drift monitoring; subgroup audits |
| **Class imbalance** | Acuity-1 severely under-represented | SMOTE + class-weight balancing (in-fold only) |
| **Age skew** | Age skew toward 40–79 (see fairness_audit output) | Age-bucket subgroup monitoring |
| **Chief-complaint vocabulary** | Site-specific phrasing; cross-fitted encoding | Review encoding map when deploying to new site |
| **Missing-not-at-random** | Vital missingness may correlate with severity | MICE imputation; `imputed_fields` flagged in API output |

---

## Consent Status

De-identified retrospective data; IRB approval assumed. Confirm with your institution's data governance office before use.

---

## PII Handling

| Field | Treatment |
|---|---|
| `chief_complaint` | Regex PII-scrubbed (`src/api/pii_scrubber.py`) before any LLM call |
| `encounter_id` | SHA-256 hashed before audit log entry; never stored raw |
| Raw vitals | Never stored in audit log or LLM prompt |
| Patient name / DOB | Not collected; not present in this dataset |

---

## Preprocessing Applied

| Step | Detail |
|---|---|
| VT#### removal | Rows with no vitals dropped at load time |
| Sensor artifact removal | Values outside physiological bounds → NaN |
| SpO2 capping | Values > 100% capped to 100% |
| MICE imputation | `IterativeImputer(max_iter=10, random_state=42)` on continuous vitals |
| Class imbalance | SMOTE applied inside training folds only |
