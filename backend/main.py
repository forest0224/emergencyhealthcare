"""FastAPI application, operations, and triage endpoints."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.security import TokenInfo, require_role
from backend.security import append_audit, hash_patient_id
from backend.reasoning import rule_based_rationale
from backend.security import scrub_free_text
from backend.reasoning import ClinicalReasonerError
from backend.schemas import ErrorOut, PatientIn, PredictionOut, RationaleOut

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool

from ml.predict import TriageModel
from backend.reasoning import ClinicalReasoner

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup (structured JSON to stdout)
# ---------------------------------------------------------------------------
import json as _json
import sys as _sys

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects.

    Example:
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(_JsonFormatter())
    """
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            # Never include stack trace in payload (PHI risk)
        return _json.dumps(payload)


def _configure_logging() -> None:
    """Configure root logger with structured JSON output.

    Example:
        >>> _configure_logging()
    """
    root = logging.getLogger()
    root.setLevel(_LOG_LEVEL)
    handler = logging.StreamHandler(_sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)


_configure_logging()
_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: load model + reasoner at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager — loads model on startup.

    Args:
        app: FastAPI application instance.

    Example:
        >>> app = FastAPI(lifespan=lifespan)
    """
    model_dir = os.environ.get("MODEL_DIR", "models")
    set_model(None)
    set_reasoner(None)

    # Load ML model
    try:
        tm = TriageModel(model_dir=model_dir)
        set_model(tm)
        _logger.info("startup model_loaded model_dir=%s", model_dir)
    except Exception as exc:
        _logger.error("startup model_load_failed error=%s", type(exc).__name__)
        # Continue — liveness will pass but readiness will fail

    # Load LLM reasoner
    try:
        cr = ClinicalReasoner()
        set_reasoner(cr)
        _logger.info("startup openai_configured=%s", cr.is_available)
    except Exception as exc:
        _logger.warning("startup reasoner_load_failed error=%s", type(exc).__name__)

    yield

    _logger.info("shutdown complete")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.

    Example:
        >>> app = create_app()
    """
    app = FastAPI(
        title       ="ED-Triage Clinical Decision Support API",
        description ="ML + LLM-powered emergency triage acuity prediction.",
        version     ="1.0.0",
        lifespan    =lifespan,
    )

    # ---- Request-ID middleware ----
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        """Inject a UUID request_id into every request and response.

        Args:
            request:   Incoming FastAPI Request.
            call_next: Next middleware / route handler.

        Returns:
            Response with X-Request-ID header.

        Example:
            >>> # Automatically applied to all requests
        """
        req_id = str(uuid.uuid4())
        request.state.request_id = req_id
        if request.method == "POST":
            try:
                size = int(request.headers.get("Content-Length", "0"))
            except ValueError:
                size = -1
            if not 0 <= size <= 16384:
                return JSONResponse(status_code=413, content=ErrorOut(
                    error_code="REQUEST_SIZE", message="Request exceeds the permitted size.",
                    request_id=req_id,
                ).model_dump(), headers={"X-Request-ID": req_id})
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response

    # ---- Global exception handler ----
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Catch all unhandled exceptions — never leak PHI or stack traces.

        Args:
            request: Incoming request.
            exc:     Unhandled exception.

        Returns:
            500 JSON error response.

        Example:
            >>> # Automatically applied to all unhandled exceptions
        """
        req_id = getattr(request.state, "request_id", "unknown")
        # Log internally with type only (not message — may contain PHI)
        _logger.error(
            "unhandled_exception request_id=%s exc_type=%s",
            req_id, type(exc).__name__,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorOut(
                error_code ="INTERNAL_ERROR",
                message    ="An internal error occurred. Please contact support.",
                request_id =req_id,
            ).model_dump(),
        )

    # ---- Validation error handler ----
    from fastapi.exceptions import RequestValidationError
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Return a clean 422 for Pydantic validation failures.

        Args:
            request: Incoming request.
            exc:     Validation error.

        Returns:
            422 JSON error response.

        Example:
            >>> # Automatically applied to all validation errors
        """
        req_id = getattr(request.state, "request_id", "unknown")
        # Extract first error message only (field names safe; values may be PHI)
        errors = exc.errors()
        first  = errors[0] if errors else {}
        field  = " → ".join(str(p) for p in first.get("loc", []))
        msg    = first.get("msg", "Validation error")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorOut(
                error_code ="VALIDATION_ERROR",
                message    =f"{field}: {msg}" if field else msg,
                request_id =req_id,
            ).model_dump(),
        )

    # ---- Mount router ----
    app.include_router(router)

    return app


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
        >>> set_reasoner(ClinicalReasoner())
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
        "reasoner":   bool(getattr(_reasoner, "is_available", False)),
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
        model_ver = _model_version()
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

    ml_out, shap_raw = await run_in_threadpool(_predict_and_explain, patient_dict)

    # proba as dict {"1":p1, "2":p2, "3":p3, "4":p4}
    raw_proba = ml_out.get("proba", [0, 0, 0, 0])
    if isinstance(raw_proba, list):
        proba_dict = {str(i + 1): float(v) for i, v in enumerate(raw_proba)}
    else:
        proba_dict = {str(k): float(v) for k, v in raw_proba.items()}

    flags     = _build_safety_flags(ml_out)
    model_ver = _model_version()

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
        latency_ms     =latency_ms,
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

    ml_out, shap_raw = await run_in_threadpool(_predict_and_explain, patient_dict)

    # proba as dict {"1":p1, ..., "4":p4}
    raw_proba = ml_out.get("proba", [0, 0, 0, 0])
    if isinstance(raw_proba, list):
        proba_dict = {str(i + 1): float(v) for i, v in enumerate(raw_proba)}
    else:
        proba_dict = {str(k): float(v) for k, v in raw_proba.items()}

    flags     = _build_safety_flags(ml_out)
    model_ver = _model_version()

    # LLM call with 5-second timeout; any failure → rule-based fallback
    ml_out["safety_flags"] = flags
    reasoner = _reasoner

    def _run_llm():
        return reasoner.reason(patient_dict, ml_out, shap_raw, request_id)

    llm_result: dict | None = None
    if reasoner is not None and getattr(reasoner, "is_available", True):
        try:
            llm_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, _run_llm),
                timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "5")),
            )
        except asyncio.TimeoutError:
            _logger.warning(
                "request_id=%s llm_timeout after %.1fs — using rule_based",
                request_id, float(os.getenv("LLM_TIMEOUT_SECONDS", "5")),
            )
        except ClinicalReasonerError as exc:
            _logger.warning(
                "request_id=%s llm_reasoner_error=%s — using rule_based",
                request_id, type(exc).__name__,
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
        latency_ms        =latency_ms,
        rationale         =llm_result.get("rationale", ""),
        reasoning_model   =llm_result.get("reasoning_model"),
        differential_risks=llm_result.get("differential_risks", []),
        abnormal_vitals   =llm_result.get("abnormal_vitals", []),
        immediate_actions =llm_result.get("immediate_actions", []),
        red_flags         =llm_result.get("red_flags", []),
        safety_note       =llm_result.get("safety_note", ""),
        llm_backend       =llm_result.get("llm_backend", "rule_based"),
    )


def _predict_and_explain(patient: dict) -> tuple[dict, dict]:
    model = _triage_model
    return model.predict_proba(patient), model.explain(patient, top_k=5)


def _model_version() -> str:
    meta = _triage_model._meta
    return meta.get("version", meta.get("best_model", "unknown"))


@router.get("/v1/status", tags=["ops"])
def system_status() -> dict:
    model = _triage_model
    return {
        "api_version": "1.0.0",
        "model_ready": model is not None,
        "model_version": _model_version() if model else None,
        "algorithm": model._meta.get("best_model") if model else None,
        "dataset_rows": model._meta.get("dataset_rows") if model else None,
        "acuity_classes": [1, 2, 3, 4],
        "openai_configured": bool(getattr(_reasoner, "is_available", False)),
        "openai_model": getattr(_reasoner, "model_name", None),
        "local_fallback_available": True,
    }


@router.get("/v1/performance", tags=["ops"])
def performance(token: TokenInfo = Depends(require_role("infer"))) -> dict:
    if _triage_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return _triage_model._meta.get("evaluation", {})


app = create_app()
