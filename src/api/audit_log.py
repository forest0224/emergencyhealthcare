"""
src/api/audit_log.py
=====================
Append-only JSONL audit trail for the ED-Triage API.

Each record captures: timestamp, request_id, token_id, patient_hash,
model_version, prediction, confidence, llm_backend, latency_ms,
imputed_fields.  Raw vitals and PHI are NEVER stored.

Output path: outputs/audit/YYYY-MM-DD.jsonl (one file per day, UTC).

Public API
----------
append_audit(record: dict) -> None

Usage
-----
>>> from src.api.audit_log import append_audit
>>> append_audit({
...     "request_id": "req-001",
...     "token_id": "tok-abc",
...     "patient_hash": "sha256...",
...     "model_version": "v1",
...     "prediction": 2,
...     "confidence": 0.72,
...     "llm_backend": "watsonx",
...     "latency_ms": 142,
...     "imputed_fields": ["spo2"],
... })
"""

from __future__ import annotations

try:
    import fcntl as _fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False  # Windows — best-effort write without flock
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_AUDIT_ROOT = Path(os.environ.get("AUDIT_LOG_DIR", "outputs/audit"))

# Fields that are NEVER allowed in audit records (safety guardrail)
_FORBIDDEN_FIELDS = frozenset([
    "age", "gender", "chief_complaint", "arrival_hour",
    "systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate",
    "temperature", "spo2", "pain_score", "encounter_id",
])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hash_patient_id(encounter_id: Optional[str]) -> str:
    """Return a SHA-256 hex digest of an encounter_id for audit linkage.

    Args:
        encounter_id: Raw encounter identifier string (may be None).

    Returns:
        16-char hex prefix of SHA-256 digest, or "unknown" if None.

    Example:
        >>> h = hash_patient_id("ENC00001")
        >>> len(h)
        16
    """
    if not encounter_id:
        return "unknown"
    return hashlib.sha256(encounter_id.encode()).hexdigest()[:16]


def append_audit(record: dict) -> None:
    """Append one audit record to today's JSONL file (UTC).

    The record is checked for forbidden fields (raw vitals / PHI) before
    writing.  A timestamp is added automatically.  Writes are serialised
    per-process via flock on POSIX; on Windows falls back to a best-effort
    write.

    Args:
        record: Dict containing audit fields. Must NOT contain raw vitals
                or PHI.  Required keys: request_id, token_id, patient_hash,
                model_version, prediction, confidence.

    Example:
        >>> append_audit({
        ...     "request_id": "req-001", "token_id": "tok-1",
        ...     "patient_hash": "abc123", "model_version": "v1",
        ...     "prediction": 2, "confidence": 0.72,
        ...     "llm_backend": "rule_based", "latency_ms": 85,
        ...     "imputed_fields": [],
        ... })
    """
    # Safety check: strip forbidden fields
    cleaned = {k: v for k, v in record.items() if k not in _FORBIDDEN_FIELDS}
    forbidden_found = set(record.keys()) & _FORBIDDEN_FIELDS
    if forbidden_found:
        _logger.warning(
            "audit: stripped forbidden fields=%s from record request_id=%s",
            sorted(forbidden_found),
            record.get("request_id", "unknown"),
        )

    # Add timestamp
    cleaned["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

    # Daily file path
    date_str  = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    _AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path  = _AUDIT_ROOT / f"{date_str}.jsonl"

    line = json.dumps(cleaned, default=str) + "\n"

    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            if _HAS_FCNTL:
                try:
                    _fcntl.flock(fh, _fcntl.LOCK_EX)
                except OSError:
                    pass
            fh.write(line)
    except Exception as exc:
        _logger.error("audit write failed: %s", exc)
