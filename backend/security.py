"""
backend/security.py
====================
Authentication, audit logging, and PII scrubbing for the ED-Triage API.

"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_logger = logging.getLogger(__name__)
_pii_logger = logging.getLogger("pii_audit")


# ═══════════════════════════════════════════════════════════════════════════
# Authentication
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TokenInfo:
    """Parsed token metadata."""
    token_id: str
    role: str
    raw: str = field(repr=False)


def _load_token_store() -> dict[str, TokenInfo]:
    """Build the in-memory token → TokenInfo map from environment."""
    store: dict[str, TokenInfo] = {}
    raw_tokens = os.environ.get("API_TOKENS", "")
    raw_admins = os.environ.get("ADMIN_TOKENS", "")
    admin_set = {t.strip() for t in raw_admins.split(",") if t.strip()}

    for tok in raw_tokens.split(","):
        tok = tok.strip()
        if not tok:
            continue
        role = "admin" if tok in admin_set else "infer"
        store[tok] = TokenInfo(token_id=hashlib.sha256(tok.encode()).hexdigest()[:12], role=role, raw=tok)

    for tok in admin_set:
        if tok and tok not in store:
            store[tok] = TokenInfo(token_id=hashlib.sha256(tok.encode()).hexdigest()[:12], role="admin", raw=tok)

    return store


_TOKEN_STORE: dict[str, TokenInfo] = {}
_security = HTTPBearer(auto_error=False)


async def get_current_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> TokenInfo:
    """Extract and validate a Bearer token from the Authorization header."""
    if credentials is None or not credentials.credentials:
        _logger.warning(
            "auth_failed reason=missing_token request_id=%s path=%s",
            getattr(request.state, "request_id", "unknown"),
            request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    global _TOKEN_STORE
    if not _TOKEN_STORE:
        _TOKEN_STORE = _load_token_store()

    token = credentials.credentials
    info = _TOKEN_STORE.get(token)

    if info is None:
        _logger.warning(
            "auth_failed reason=invalid_token request_id=%s path=%s",
            getattr(request.state, "request_id", "unknown"),
            request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    _logger.info(
        "auth_ok token_id=%s role=%s path=%s request_id=%s",
        info.token_id, info.role, request.url.path,
        getattr(request.state, "request_id", "unknown"),
    )
    return info


def require_role(role: str) -> Callable:
    """Return a FastAPI dependency that enforces a minimum role."""
    _ROLE_ORDER = {"infer": 0, "admin": 1}

    async def _dependency(
        token_info: TokenInfo = Depends(get_current_token),
    ) -> TokenInfo:
        required_level = _ROLE_ORDER.get(role, 0)
        actual_level = _ROLE_ORDER.get(token_info.role, 0)
        if actual_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required; caller has '{token_info.role}'.",
            )
        return token_info

    return _dependency


# ═══════════════════════════════════════════════════════════════════════════
# Audit Logging
# ═══════════════════════════════════════════════════════════════════════════

_AUDIT_ROOT = Path(os.environ.get("AUDIT_LOG_DIR", "outputs/audit"))

_FORBIDDEN_FIELDS = frozenset([
    "age", "gender", "chief_complaint", "arrival_hour",
    "systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate",
    "temperature", "spo2", "pain_score", "encounter_id",
])

try:
    import fcntl as _fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


def hash_patient_id(encounter_id: Optional[str]) -> str:
    """Return a SHA-256 hex prefix of an encounter_id for audit linkage."""
    if not encounter_id:
        return "unknown"
    return hashlib.sha256(encounter_id.encode()).hexdigest()[:16]


def append_audit(record: dict) -> None:
    """Append one audit record to today's JSONL file (UTC)."""
    cleaned = {k: v for k, v in record.items() if k not in _FORBIDDEN_FIELDS}
    forbidden_found = set(record.keys()) & _FORBIDDEN_FIELDS
    if forbidden_found:
        _logger.warning(
            "audit: stripped forbidden fields=%s from record request_id=%s",
            sorted(forbidden_found),
            record.get("request_id", "unknown"),
        )

    cleaned["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    _AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = _AUDIT_ROOT / f"{date_str}.jsonl"
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


# ═══════════════════════════════════════════════════════════════════════════
# PII Scrubber
# ═══════════════════════════════════════════════════════════════════════════

_MRN_PATTERN = re.compile(
    r"\b(?:MRN[:\s#]*|mrn[:\s#]*)?\d{5,10}\b", re.IGNORECASE,
)
_PHONE_PATTERN = re.compile(
    r"\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"
)
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
    r"|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b",
    re.IGNORECASE,
)
_NAME_PATTERN = re.compile(
    r"\b(?:Dr\.?|Doctor|Nurse|RN|Mr\.?|Mrs\.?|Ms\.?|Prof\.?)\s+[A-Z][a-z]+"
    r"(?:\s+[A-Z][a-z]+)?",
    re.IGNORECASE,
)

_RULES: list[tuple[re.Pattern, str]] = [
    (_NAME_PATTERN, "[NAME]"),
    (_EMAIL_PATTERN, "[EMAIL]"),
    (_PHONE_PATTERN, "[PHONE]"),
    (_MRN_PATTERN, "[MRN]"),
    (_DATE_PATTERN, "[DATE]"),
]


def scrub_free_text(text: str) -> str:
    """Remove PII/PHI from free-text clinical input."""
    if not isinstance(text, str):
        return str(text)

    result = text
    for pattern, tag in _RULES:
        def _replace(m: re.Match, _tag: str = tag) -> str:
            _log_redaction(m.group(0), _tag)
            return _tag
        result = pattern.sub(_replace, result)
    return result


def _log_redaction(raw: str, tag: str) -> None:
    """Log a SHA-256 hash of a redacted span — never the raw value."""
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    _pii_logger.info("pii_redacted tag=%s hash=%s", tag, digest)
