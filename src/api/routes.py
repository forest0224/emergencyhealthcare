"""
src/api/routes.py
==================
FastAPI router for the ED-Triage Clinical Decision Support API.

Endpoints
---------
POST /v1/predict   – ML-only fast path (no LLM)
POST /v1/triage    – Full path: ML + SHAP + LLM rationale
GET  /v1/health    – Liveness probe
GET  /v1/ready     – Readiness probe (model + LLM reachable)
GET  /v1/version   – Model + API version info
GET  /v1/metrics   – Prometheus metrics

Usage
-----
Imported by src/api/main.py via app.include_router(router).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.auth import TokenInfo, get_current_token, require_role
from src.api.audit_log import append_audit, hash_patient_id
from src.api.fallback_rationale import rule_based_rationale
from src.api.pii_scrubber import scrub_free_text
from src.api.clinical_reasoning_llm import ClinicalReasonerError
from src.api.schemas import ErrorOut, PatientIn, PredictionOut, RationaleOut

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: Prometheus metrics
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from starlette.responses import Response as _StarletteResponse

    _REQUEST_COUNT = Counter(
        "triage_requests_total",
        "Total triage API requests",
        ["endpoint", "status"],
    )
    _LATENCY = Histogram(
        "triage_request_latency_seconds",
        "Request latency in seconds",
        ["endpoint"],
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    _HAS_PROM = True
except ImportError:
    _HAS_PROM = False

# ---------------------------------------------------------------------------
# Module-level model instance (loaded once at startup via lifespan)
# ---------------------------------------------------------------------------
_triage_model: Any = None
_reasoner:     Any = None

LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT_SECONDS", "5"))


def get_model():
    """Return the singleton TriageModel instance.

    Returns:
        Loaded TriageModel or None.

    Example:
        >>> m = get_model()
    """
    return _triage_model


def get_reasoner():
    """Return the singleton ClinicalReasoner instance.

    Returns:
        Loaded ClinicalReasoner or None.

    Example:
        >>> r = get_reasoner()
    """
    return _reasoner


def set_model(m: Any) -> None:
    """Set the module-level TriageModel (called from lifespan).

    Args:
        m: Fitted TriageModel instance.

    Example:
        >>> set_model(TriageModel("ml_runs"))
    """
    global _triage_model
    _triage_model = m


def set_reasoner(r: Any) -> None:
    """Set the module-level ClinicalReasoner (called from lifespan).

    Args:
        r: ClinicalReasoner instance.

    Example:
        >>> set_reasoner(ClinicalReasoner("watsonx"))
    """
    global _reasoner
    _reasoner = r


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter()


def _build_safety_flags(ml_out: dict) -> list[str]:
    """Derive safety flags from ML output.

    Args:
        ml_out: Output dict from TriageModel.predict_proba().

    Returns:
        List of safety flag strings.

    Example:
        >>> flags = _build_safety_flags({"predicted": 1, "confidence": 0.55})
    """
    flags: list[str] = []
    proba = ml_out.get("proba", [0, 0, 0, 0])
    if len(proba) >= 2:
        ha_prob = proba[0] + proba[1]
        if ha_prob >= 0.30 and ml_out.get("predicted", 4) >= 3:
            flags.append("high_acuity_probability_despite_low_prediction")
    if ml_out.get("confidence", 1.0) < 0.40:
        flags.append("low_confidence_prediction")
    if ml_out.get("predicted") == 1:
        flags.append("critical_acuity_immediate_review_required")
    return flags


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/v1/health", tags=["ops"])
async def health() -> dict:
    """Liveness probe — returns 200 if the process is alive.

    Returns:
        Dict with status 'ok'.

    Example:
        >>> GET /v1/health -> {"status": "ok"}
    """
    return {"status": "ok"}


@router.get("/v1/ready", tags=["ops"])
async def ready() -> dict:
    """Readiness probe — checks that the model is loaded.

    Returns:
        Dict with status and component health.

    Raises:
        HTTPException 503: If model is not loaded.

    Example:
        >>> GET /v1/ready -> {"status": "ready", "model": true, ...}
    """
    model_ok = _triage_model is not None
    if not model_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )
    return {
        "status":     "ready",
        "model":      model_ok,
        "reasoner":   _reasoner is not None,
    }


@router.get("/v1/version", tags=["ops"])
async def version() -> dict:
    """Return model and API version information.

    Returns:
        Dict with api_version and model info.

    Example:
        >>> GET /v1/version -> {"api_version": "1.0.0", ...}
    """
    model_ver = "unknown"
    if _triage_model is not None:
        model_ver = _triage_model._meta.get("best_model", "unknown")
    return {
        "api_version":   "1.0.0",
        "model_version": model_ver,
    }


@router.get("/v1/metrics", tags=["ops"])
async def metrics():
    """Expose Prometheus metrics.

    Returns:
        Plain-text Prometheus metrics or 501 if unavailable.

    Example:
        >>> GET /v1/metrics
    """
    if not _HAS_PROM:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="prometheus-client not installed.",
        )
    return _StarletteResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.post("/v1/predict", response_model=PredictionOut, tags=["inference"])
async def predict(
    request:    Request,
    patient:    PatientIn,
    token_info: TokenInfo = Depends(require_role("infer")),
) -> PredictionOut:
    """ML-only fast-path prediction endpoint.

    Does not invoke the LLM.  Returns probability, predicted acuity,
    confidence, imputed fields, top SHAP features, and safety flags.

    Args:
        request:    FastAPI Request (for request_id + timing).
        patient:    Validated PatientIn payload.
        token_info: Authenticated token (requires 'infer' role).

    Returns:
        PredictionOut JSON.

    Example:
        >>> POST /v1/predict {"age": 65, "heart_rate": 120, ...}
    """
    t0         = time.perf_counter()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    if _triage_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )

    patient_dict = patient.model_dump()
    # Scrub chief complaint
    patient_dict["chief_complaint"] = scrub_free_text(
        patient_dict.get("chief_complaint", "")
    )

    ml_out   = _triage_model.predict_proba(patient_dict)
    shap_raw = _triage_model.explain(patient_dict, top_k=5)

    # proba as dict {"1":p1, "2":p2, "3":p3, "4":p4}
    raw_proba = ml_out.get("proba", [0, 0, 0, 0])
    if isinstance(raw_proba, list):
        proba_dict = {str(i + 1): float(v) for i, v in enumerate(raw_proba)}
    else:
        proba_dict = {str(k): float(v) for k, v in raw_proba.items()}

    flags     = _build_safety_flags(ml_out)
    model_ver = _triage_model._meta.get("best_model", "unknown")

    latency_ms = int((time.perf_counter() - t0) * 1000)

    append_audit({
        "request_id":    request_id,
        "token_id":      token_info.token_id,
        "patient_hash":  hash_patient_id(patient_dict.get("encounter_id")),
        "model_version": model_ver,
        "prediction":    ml_out["predicted"],
        "confidence":    ml_out["confidence"],
        "llm_backend":   "none",
        "latency_ms":    latency_ms,
        "imputed_fields": ml_out.get("imputed_fields", []),
    })

    if _HAS_PROM:
        _REQUEST_COUNT.labels(endpoint="/v1/predict", status="200").inc()
        _LATENCY.labels(endpoint="/v1/predict").observe(latency_ms / 1000)

    return PredictionOut(
        proba          =proba_dict,
        predicted      =ml_out["predicted"],
        confidence     =ml_out["confidence"],
        imputed_fields =ml_out.get("imputed_fields", []),
        shap_top       =shap_raw,
        safety_flags   =flags,
        model_version  =model_ver,
    )


@router.post("/v1/triage", response_model=RationaleOut, tags=["inference"])
async def triage(
    request:    Request,
    patient:    PatientIn,
    token_info: TokenInfo = Depends(require_role("infer")),
) -> RationaleOut:
    """Full triage endpoint: ML + SHAP + LLM clinical rationale.

    Calls the LLM with a 5-second timeout.  On timeout or failure,
    falls back to rule_based_rationale() — always returns 200.

    Args:
        request:    FastAPI Request.
        patient:    Validated PatientIn payload.
        token_info: Authenticated token (requires 'infer' role).

    Returns:
        RationaleOut JSON.

    Example:
        >>> POST /v1/triage {"age": 70, "systolic_bp": 80, ...}
    """
    t0         = time.perf_counter()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    if _triage_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )

    patient_dict = patient.model_dump()
    patient_dict["chief_complaint"] = scrub_free_text(
        patient_dict.get("chief_complaint", "")
    )

    ml_out   = _triage_model.predict_proba(patient_dict)
    shap_raw = _triage_model.explain(patient_dict, top_k=5)

    # proba as dict {"1":p1, ..., "4":p4}
    raw_proba = ml_out.get("proba", [0, 0, 0, 0])
    if isinstance(raw_proba, list):
        proba_dict = {str(i + 1): float(v) for i, v in enumerate(raw_proba)}
    else:
        proba_dict = {str(k): float(v) for k, v in raw_proba.items()}

    flags     = _build_safety_flags(ml_out)
    model_ver = _triage_model._meta.get("best_model", "unknown")

    # LLM call with 5-second timeout; any failure → rule-based fallback
    def _run_llm():
        return _reasoner.reason(patient_dict, ml_out, shap_raw, request_id)

    llm_result: dict | None = None
    if _reasoner is not None:
        try:
            llm_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, _run_llm),
                timeout=LLM_TIMEOUT,
            )
        except asyncio.TimeoutError:
            _logger.warning(
                "request_id=%s llm_timeout after %.1fs — using rule_based",
                request_id, LLM_TIMEOUT,
            )
        except ClinicalReasonerError as exc:
            _logger.warning(
                "request_id=%s llm_reasoner_error=%s — using rule_based",
                request_id, exc,
            )
        except Exception as exc:
            _logger.error(
                "request_id=%s llm_unexpected=%s — using rule_based",
                request_id, type(exc).__name__,
            )

    if llm_result is None:
        llm_result = rule_based_rationale(patient_dict, ml_out)

    latency_ms = int((time.perf_counter() - t0) * 1000)

    append_audit({
        "request_id":    request_id,
        "token_id":      token_info.token_id,
        "patient_hash":  hash_patient_id(patient_dict.get("encounter_id")),
        "model_version": model_ver,
        "prediction":    ml_out["predicted"],
        "confidence":    ml_out["confidence"],
        "llm_backend":   llm_result.get("llm_backend", "unknown"),
        "latency_ms":    latency_ms,
        "imputed_fields": ml_out.get("imputed_fields", []),
    })

    if _HAS_PROM:
        _REQUEST_COUNT.labels(endpoint="/v1/triage", status="200").inc()
        _LATENCY.labels(endpoint="/v1/triage").observe(latency_ms / 1000)

    return RationaleOut(
        predicted_acuity  =ml_out["predicted"],
        confidence        =ml_out["confidence"],
        proba             =proba_dict,
        imputed_fields    =ml_out.get("imputed_fields", []),
        shap_top          =shap_raw,
        safety_flags      =flags,
        model_version     =model_ver,
        differential_risks=llm_result.get("differential_risks", []),
        abnormal_vitals   =llm_result.get("abnormal_vitals", []),
        immediate_actions =llm_result.get("immediate_actions", []),
        red_flags         =llm_result.get("red_flags", []),
        safety_note       =llm_result.get("safety_note", ""),
        llm_backend       =llm_result.get("llm_backend", "rule_based"),
    )
