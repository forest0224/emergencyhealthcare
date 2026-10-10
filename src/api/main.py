"""
src/api/main.py
================
FastAPI application factory for the ED-Triage Clinical Decision Support API.

Responsibilities:
  - Request-ID middleware (UUID injected into every request state + response header)
  - Structured JSON logging
  - Global exception handler (never leaks PHI or stack traces to clients)
  - Lifespan startup: load TriageModel + ClinicalReasoner
  - Mount router from routes.py

Usage
-----
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2
"""

from __future__ import annotations

import logging
import os
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from src.api.routes import router, set_model, set_reasoner
from src.api.schemas import ErrorOut

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
    model_dir  = os.environ.get("MODEL_DIR", "ml_runs")
    llm_backend = os.environ.get("LLM_BACKEND", "watsonx")

    # Load ML model
    try:
        from src.ml.inference_api import TriageModel
        tm = TriageModel(model_dir=model_dir)
        set_model(tm)
        _logger.info("startup model_loaded model_dir=%s", model_dir)
    except Exception as exc:
        _logger.error("startup model_load_failed error=%s", exc)
        # Continue — liveness will pass but readiness will fail

    # Load LLM reasoner
    try:
        from src.api.clinical_reasoning_llm import ClinicalReasoner
        cr = ClinicalReasoner(backend=llm_backend)
        set_reasoner(cr)
        _logger.info("startup reasoner_loaded backend=%s", llm_backend)
    except Exception as exc:
        _logger.warning("startup reasoner_load_failed error=%s", exc)

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
        docs_url    =None,
        redoc_url   =None,
        openapi_url =None,
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
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = req_id
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


# Module-level app instance (used by uvicorn)
app = create_app()
