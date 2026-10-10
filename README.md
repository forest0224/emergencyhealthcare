# 🏥 Emergency Healthcare — ED-Triage Clinical Decision Support

> **Decision-support only. Not a diagnosis. Confirm with a licensed clinician.**

An end-to-end ML + LLM system that predicts Emergency Department triage acuity (ESI levels 1–4), explains predictions with SHAP, and generates clinical rationale via IBM watsonx.ai or OpenAI — served through a production-ready FastAPI backend and a Python Tkinter desktop GUI.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Python GUI Interface](#python-gui-interface)
- [API Reference](#api-reference)
- [ML Pipeline](#ml-pipeline)
- [Docker Deployment](#docker-deployment)
- [Testing](#testing)
- [Governance & Safety](#governance--safety)
- [Tech Stack](#tech-stack)

---

## Features

- 🤖 **XGBoost / LightGBM / CatBoost** ensemble trained on emergency triage data
- 🔍 **SHAP explanations** — top feature contributions per acuity class
- 🧠 **LLM clinical rationale** via IBM watsonx.ai (`granite-13b-chat-v2`) or OpenAI (`gpt-4o-mini`)
- 🛡️ **Rule-based fallback** — always returns a safe response even when LLM is unavailable
- 🖥️ **Python Tkinter desktop GUI** — submit patients and view results interactively
- 📊 **Prometheus + Grafana** observability stack
- 🔐 **Bearer token auth** with role-based access (`infer` / `admin`)
- 📋 **Audit logging** with SHA-256 patient ID hashing (no PHI stored)
- ⚖️ **Fairness audit** across age/gender groups
- 📦 **Docker Compose** one-command deployment

---

## Project Structure

```
emergencyhealthcare/
├── interface.py                  # Python Tkinter desktop GUI
├── run_full_pipeline.py          # CLI entry point (train / infer / demo / serve / audit)
├── demo_patient.json             # Sample patient for quick testing
├── master_emergency_triage_dataset.csv  # Training dataset
│
├── src/
│   ├── api/
│   │   ├── main.py               # FastAPI app factory + lifespan
│   │   ├── routes.py             # Endpoints: /v1/predict, /v1/triage, /v1/health …
│   │   ├── schemas.py            # Pydantic v2 request/response models
│   │   ├── auth.py               # Bearer token authentication
│   │   ├── audit_log.py          # Append-only JSONL audit trail
│   │   ├── clinical_reasoning_llm.py  # watsonx / OpenAI LLM client
│   │   ├── fallback_rationale.py # Rule-based clinical rationale (no LLM)
│   │   └── pii_scrubber.py       # Free-text PII scrubbing before LLM calls
│   └── ml/
│       ├── feature_engineering.py  # MICE imputation + feature construction
│       ├── tabular_models.py        # Train XGBoost / LightGBM / CatBoost
│       ├── inference_api.py         # TriageModel — predict + SHAP explain
│       ├── calibration.py           # Probability calibration
│       ├── safety_metrics.py        # Critical miss rate + safety KPIs
│       ├── fairness_audit.py        # Demographic parity / equalized odds
│       ├── drift_monitor.py         # Feature drift detection (KS test)
│       └── model_registry.py        # Model versioning helpers
│
├── tests/                        # pytest test suite
├── docs/
│   ├── MODEL_CARD.md
│   ├── DATA_CARD.md
│   └── DEPLOYMENT_RUNBOOK.md
├── docker-compose.yml            # API + Prometheus + Grafana
├── Dockerfile
├── prometheus.yml
├── Makefile
└── requirements.txt
```

---

## Quick Start

### 1. Clone & set up environment

```bash
git clone <repo-url>
cd emergencyhealthcare

python -m venv .venv311
# Windows
.venv311\Scripts\activate
# macOS / Linux
source .venv311/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys (see Configuration below)
```

### 3. Train the model

```bash
python run_full_pipeline.py --mode train
```

### 4. Start the API server

```bash
python run_full_pipeline.py --mode serve
# Server starts at http://localhost:8000
```

### 5. Launch the desktop GUI

```bash
python interface.py
```

---

## Configuration

Copy `.env.example` to `.env` and fill in the values:

| Variable | Description | Default |
|---|---|---|
| `WATSONX_API_KEY` | IBM watsonx.ai API key | *(required for watsonx backend)* |
| `WATSONX_PROJECT_ID` | watsonx project ID | *(required for watsonx backend)* |
| `WATSONX_URL` | watsonx endpoint URL | `https://us-south.ml.cloud.ibm.com` |
| `WATSONX_MODEL_ID` | LLM model ID | `ibm/granite-13b-chat-v2` |
| `OPENAI_API_KEY` | OpenAI API key | *(required for openai backend)* |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` |
| `BACKEND` | LLM backend selection | `watsonx` |
| `LLM_TIMEOUT_SECONDS` | LLM call timeout | `5` |
| `API_TOKENS` | Comma-separated Bearer tokens | `dev-token-1,dev-token-2` |
| `ADMIN_TOKENS` | Admin-role tokens (subset of API_TOKENS) | *(empty)* |
| `MODEL_DIR` | Directory for saved models | `ml_runs` |
| `AUDIT_LOG_DIR` | Audit log output directory | `outputs/audit` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

> **Tip:** The server works without LLM keys — it falls back to the rule-based rationale engine automatically.

---

## Running the Project

### CLI modes (`run_full_pipeline.py`)

| Command | Description |
|---|---|
| `python run_full_pipeline.py --mode train` | Train & evaluate ML models, save best model |
| `python run_full_pipeline.py --mode demo` | Run 3 canned patients through the full pipeline |
| `python run_full_pipeline.py --mode infer --patient demo_patient.json` | Single patient inference |
| `python run_full_pipeline.py --mode serve` | Start FastAPI server (port 8000) |
| `python run_full_pipeline.py --mode test` | Run the pytest test suite |
| `python run_full_pipeline.py --mode audit` | Run fairness + drift audit |

### Makefile shortcuts

```bash
make install      # pip install -r requirements.txt
make train        # train ML models
make serve        # start API server
make test         # run tests
make audit        # fairness audit
make docker-up    # start full Docker stack
make docker-down  # stop Docker stack
```

---

## Python GUI Interface

Launch with:

```bash
python interface.py
```

The GUI connects to `http://localhost:8000` (make sure the server is running first).

**Features:**
- Patient input form — all vitals, demographics, chief complaint
- Live API status indicator (auto-refreshes every 8 seconds)
- Colour-coded acuity badge: 🔴 Critical · 🟠 Emergent · 🟡 Urgent · 🟢 Less Urgent
- Probability bars per acuity class
- SHAP feature contribution viewer
- Clinical rationale: immediate actions, differential risks, red flags
- Raw JSON response tab
- Endpoint switcher: fast predict vs full triage + rationale

---

## API Reference

Base URL: `http://localhost:8000`

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/v1/health` | None | Liveness probe |
| `GET` | `/v1/ready` | None | Readiness probe (model loaded?) |
| `GET` | `/v1/version` | None | API and model version info |
| `GET` | `/v1/metrics` | None | Prometheus metrics |
| `POST` | `/v1/predict` | Bearer token | ML-only fast prediction |
| `POST` | `/v1/triage` | Bearer token | ML + SHAP + LLM rationale |

Interactive docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**

### Example request

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Authorization: Bearer dev-token-1" \
  -H "Content-Type: application/json" \
  -d @demo_patient.json
```

### PatientIn fields

| Field | Type | Required | Range | Description |
|---|---|---|---|---|
| `age` | float | ✅ | 0–120 | Age in years |
| `chief_complaint` | string | ✅ | 1–500 chars | Free-text complaint |
| `gender` | string | | M / F | Biological sex |
| `arrival_hour` | int | | 0–23 | Hour of arrival |
| `systolic_bp` | float | | 40–300 | mmHg |
| `diastolic_bp` | float | | 20–200 | mmHg |
| `heart_rate` | float | | 20–300 | bpm |
| `respiratory_rate` | float | | 4–60 | breaths/min |
| `temperature` | float | | 30–43 | °C |
| `spo2` | float | | 50–100 | % |
| `pain_score` | float | | 0–10 | Self-reported |
| `encounter_id` | string | | | Audit linkage (hashed) |

---

## ML Pipeline

1. **Feature Engineering** — MICE imputation, NEWS2 proxy score, shock index, SpO₂ deficit, temp deviation, cyclical arrival hour encoding
2. **Training** — 5-fold stratified cross-validation across XGBoost, LightGBM, and CatBoost; best model selected by macro F1
3. **Calibration** — Isotonic regression probability calibration
4. **SHAP** — TreeExplainer values computed per request for the top-k features
5. **Safety metrics** — Critical miss rate (acuity 1/2 predicted as 4), under-triage rate
6. **Drift monitoring** — KS-test on reference distribution (fit at train time)

---

## Docker Deployment

```bash
# Build and start API + Prometheus + Grafana
docker-compose up -d

# Services:
#   API        → http://localhost:8000
#   Prometheus → http://localhost:9090
#   Grafana    → http://localhost:3000  (admin / changeme_grafana_admin)
```

---

## Testing

```bash
pytest -q
# or
python run_full_pipeline.py --mode test
```

Test coverage includes API endpoints, schema validation, ML pipeline, and inference.

---

## Governance & Safety

- **Audit log** — every prediction is logged (request ID, token ID, hashed patient ID, prediction, confidence, latency). No raw PHI stored.
- **PII scrubbing** — chief complaint is scrubbed before any LLM call.
- **Fairness audit** — demographic parity and equalized odds across age buckets and gender groups.
- **Critical miss rate** — tracked at train time and reported in `ml_runs/safety_report.json`.
- **LLM timeout** — 5-second hard timeout on all LLM calls; always falls back to rule-based rationale.
- **Disclaimer** — every API response includes: *"Decision-support only. Not a diagnosis. Confirm with a licensed clinician."*

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | XGBoost · LightGBM · CatBoost · scikit-learn |
| Explainability | SHAP |
| LLM | IBM watsonx.ai (Granite) · OpenAI |
| API | FastAPI · Uvicorn · Pydantic v2 |
| GUI | Python Tkinter (stdlib) |
| Observability | Prometheus · Grafana |
| Containerisation | Docker · Docker Compose |
| Testing | pytest · httpx |
| Python | 3.11+ |

---

> Made with ❤️ for emergency healthcare decision support.
