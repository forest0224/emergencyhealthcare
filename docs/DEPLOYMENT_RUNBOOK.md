# Deployment Runbook — ED AI Triage

> **Audience:** ML Platform Engineers, Clinical Informatics, On-Call DevOps

---

## 1. First-Time Setup

```bash
cp .env.example .env          # fill in WATSONX_* or OPENAI_* keys
make install                  # pip install -r requirements.txt
make train                    # feature engineering → model training → safety_report.json
make test                     # run pytest (must pass before serving)
```

After `make train`, review `ml_runs/safety_report.json`.
**Gate:** `critical_miss_rate` must be ≤ 0.02 before proceeding.

---

## 2. Local Serve

```bash
make serve
# → http://localhost:8000
# Docs: http://localhost:8000/docs
```

Test with the demo patient:
```bash
curl -X POST http://localhost:8000/v1/triage \
     -H "Authorization: Bearer dev-token-1" \
     -H "Content-Type: application/json" \
     -d @demo_patient.json
```

---

## 3. Docker Deploy

```bash
make docker-build             # builds ed-triage-api image
make docker-up                # starts api + prometheus + grafana
# API:       http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana:   http://localhost:3000  (admin / changeme_grafana_admin)
```

Reverse-proxy behind nginx or an ALB with TLS before exposing externally.

To stop:
```bash
make docker-down
```

---

## 4. Model Promotion

After training a new model version and verifying metrics:

```bash
python - <<'EOF'
from src.ml.model_registry import ModelRegistry
reg = ModelRegistry("ml_runs")
print(reg.list_versions()[["version_id", "stage", "macro_f1"]])
reg.promote("<version_id>")   # replace with version_id from list above
EOF

# Restart API to pick up the new model
docker-compose up -d --no-deps --build api
```

---

## 5. Rollback

```bash
python - <<'EOF'
from src.ml.model_registry import ModelRegistry
reg = ModelRegistry("ml_runs")
reg.rollback("<previous_stable_version_id>")
EOF

docker-compose up -d --no-deps api
```

**SLA target:** Rollback completed within 15 minutes of detection.

---

## 6. Drift Alert Response

Triggered when PSI > 0.20 on any feature (logged as `[ALERT]` by `drift_monitor.py`).

```bash
# Run drift check manually against a recent batch
python -c "
from src.ml.drift_monitor import check_drift
import pandas as pd
X_new = pd.read_csv('outputs/recent_batch_features.csv')
print(check_drift(X_new)[lambda d: d.psi > 0.1])
"
```

Response:
1. Identify drifted feature(s).
2. Investigate root cause (intake workflow change? seasonal shift? ETL bug?).
3. If genuine population shift → retrain within **48 hours**.
4. If ETL/data bug → fix upstream; do not retrain yet.

---

## 7. LLM Outage Response

The API remains up during any LLM outage. `ClinicalReasoner` automatically raises `ClinicalReasonerError`; the route handler catches it and calls `rule_based_rationale()` instead. Clients always receive HTTP 200 with `"llm_backend": "rule_based"`.

**No immediate action required**, but:
- Alert on-call engineer if outage > 30 minutes.
- Switch backend via `.env`: set `BACKEND=openai` (or `BACKEND=watsonx`) and restart.

```bash
# Switch LLM backend without rebuild
sed -i 's/^BACKEND=.*/BACKEND=openai/' .env
docker-compose up -d --no-deps api
```

---

## 8. Incident Escalation

| Severity | Condition | Escalation | SLA |
|---|---|---|---|
| P1 | Critical miss rate > 2% in production | ML Lead + Clinical AI Lead + CMO | 1 hour |
| P2 | PSI > 0.20 + recall drop > 10% | ML Lead + Clinical Informatics | 4 hours |
| P3 | LLM outage > 2 hours | On-call engineer | 2 hours |
| P4 | API down (503) | DevOps on-call | 30 minutes |
| P5 | Drift alert only (PSI 0.10–0.20) | ML Lead (next business day) | 24 hours |

**Escalation path:**
`On-call engineer → ML Platform Lead → Clinical Informatics Director → Compliance Officer`

---

## 9. Useful Commands

```bash
# Health check
curl http://localhost:8000/v1/health

# Model version
curl http://localhost:8000/v1/version

# Today's audit log
cat outputs/audit/$(date +%Y-%m-%d).jsonl | python -m json.tool | head -40

# List model versions
python -c "from src.ml.model_registry import ModelRegistry; \
           print(ModelRegistry('ml_runs').list_versions())"

# Fairness + drift report
make audit
```
