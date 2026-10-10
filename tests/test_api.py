"""
tests/test_api.py
==================
FastAPI integration tests using httpx + ASGITransport (Python 3.13 compatible).
"""
import asyncio
import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("API_TOKENS", "test-tok,test-admin")
os.environ.setdefault("ADMIN_TOKENS", "test-admin")
os.environ.setdefault("MODEL_DIR", "ml_runs")


# ---------------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------------

class _FakeModel:
    def __init__(self):
        self._meta = {
            "best_model": "FakeModel",
            "feature_names": [],
            "label_imap": {i: i + 1 for i in range(4)},
        }

    def predict_proba(self, patient_dict: dict) -> dict:
        return {
            "proba": [0.6, 0.25, 0.1, 0.05],
            "predicted": 1,
            "confidence": 0.6,
            "imputed_fields": ["spo2"],
            "feature_vector": {},
        }

    def explain(self, patient_dict: dict, top_k: int = 5) -> dict:
        return {"1": [{"feature": "shock_index", "shap_value": 0.42}],
                "2": [], "3": [], "4": []}


# ---------------------------------------------------------------------------
# Sync helper that wraps async httpx calls
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine synchronously, reusing the event loop if available."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class _Client:
    """Thin sync wrapper around httpx AsyncClient + ASGITransport."""
    def __init__(self, app):
        self._app = app

    def _req(self, method, url, **kwargs):
        import httpx
        async def _call():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self._app),
                base_url="http://test",
            ) as c:
                return await c.request(method, url, **kwargs)
        return _run(_call())

    def get(self, url, **kw):  return self._req("GET",  url, **kw)
    def post(self, url, **kw): return self._req("POST", url, **kw)


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from src.api.main import create_app
    import src.api.routes as _routes
    from src.api.fallback_rationale import rule_based_rationale

    _routes.set_model(_FakeModel())

    fake_reasoner = MagicMock()
    fb = rule_based_rationale({"age": 65, "heart_rate": 120}, {"predicted": 1})
    fb["llm_backend"] = "watsonx"
    fake_reasoner.reason.return_value = fb
    _routes.set_reasoner(fake_reasoner)

    return _Client(create_app())


_AUTH    = {"Authorization": "Bearer test-tok"}
_PATIENT = {
    "age": 65, "gender": "M", "chief_complaint": "chest pain",
    "arrival_hour": 14, "systolic_bp": 80, "heart_rate": 120,
    "respiratory_rate": 24, "temperature": 37.2, "spo2": 94, "pain_score": 7,
}


# ---------------------------------------------------------------------------
# /v1/predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_valid_returns_200(self, client):
        assert client.post("/v1/predict", json=_PATIENT, headers=_AUTH).status_code == 200

    def test_response_has_required_keys(self, client):
        body = client.post("/v1/predict", json=_PATIENT, headers=_AUTH).json()
        for k in ("proba", "predicted", "confidence", "disclaimer"):
            assert k in body, f"Missing: {k}"

    def test_proba_is_dict(self, client):
        proba = client.post("/v1/predict", json=_PATIENT, headers=_AUTH).json()["proba"]
        assert isinstance(proba, dict) and "1" in proba

    def test_no_auth_returns_401(self, client):
        assert client.post("/v1/predict", json=_PATIENT).status_code == 401

    def test_invalid_age_returns_422(self, client):
        assert client.post("/v1/predict", json={**_PATIENT, "age": 200},
                           headers=_AUTH).status_code == 422

    def test_invalid_spo2_returns_422(self, client):
        assert client.post("/v1/predict", json={**_PATIENT, "spo2": 150},
                           headers=_AUTH).status_code == 422


# ---------------------------------------------------------------------------
# /v1/triage
# ---------------------------------------------------------------------------

class TestTriage:
    def test_valid_returns_200(self, client):
        assert client.post("/v1/triage", json=_PATIENT, headers=_AUTH).status_code == 200

    def test_response_has_rationale_fields(self, client):
        body = client.post("/v1/triage", json=_PATIENT, headers=_AUTH).json()
        for k in ("predicted_acuity", "immediate_actions",
                  "abnormal_vitals", "differential_risks", "disclaimer"):
            assert k in body, f"Missing: {k}"

    def test_fallback_when_llm_raises(self, client):
        """LLM raising ClinicalReasonerError → rule_based_rationale used, 200."""
        import src.api.routes as _routes
        from src.api.clinical_reasoning_llm import ClinicalReasonerError

        bad = MagicMock()
        bad.reason.side_effect = ClinicalReasonerError("down")
        orig = _routes._reasoner
        _routes.set_reasoner(bad)
        try:
            resp = client.post("/v1/triage", json=_PATIENT, headers=_AUTH)
            assert resp.status_code == 200
            body = resp.json()
            assert body["llm_backend"] == "rule_based"
            assert len(body["immediate_actions"]) > 0
        finally:
            _routes.set_reasoner(orig)


# ---------------------------------------------------------------------------
# Ops endpoints
# ---------------------------------------------------------------------------

def test_health(client):
    assert client.get("/v1/health").status_code == 200

def test_version(client):
    assert "model_version" in client.get("/v1/version").json()


# ---------------------------------------------------------------------------
# PII scrubber
# ---------------------------------------------------------------------------

def test_pii_scrubber_removes_phi():
    from src.api.pii_scrubber import scrub_free_text
    result = scrub_free_text("Dr. Smith MRN 12345")
    assert "Smith" not in result
    assert "12345" not in result
