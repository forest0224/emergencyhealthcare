"""
src/api/auth.py
================
Bearer-token authentication and role-based access control for the
ED-Triage FastAPI application.

Tokens are read from the environment variable API_TOKENS (comma-separated).
Each token has an associated role:
  infer – may call /v1/predict and /v1/triage
  admin – may call all endpoints including registry operations

Every authenticated call is logged: request_id, token_id, endpoint,
latency_ms, model_version.

Public API
----------
get_current_token(credentials) -> TokenInfo
require_role(role) -> Callable

Usage
-----
>>> from src.api.auth import require_role
>>> @router.post("/v1/admin/promote")
... async def promote(dep=Depends(require_role("admin"))):
...     ...
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from dotenv import load_dotenv
load_dotenv()

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token store
# ---------------------------------------------------------------------------

@dataclass
class TokenInfo:
    """Parsed token metadata.

    Attributes:
        token_id: Short identifier (first 8 chars of token).
        role:     Access role ('infer' or 'admin').
        raw:      Full token string (kept in memory only, never logged).

    Example:
        >>> ti = TokenInfo(token_id="abc12345", role="infer", raw="...")
    """
    token_id: str
    role:     str
    raw:      str = field(repr=False)


def _load_token_store() -> dict[str, TokenInfo]:
    """Build the in-memory token → TokenInfo map from environment.

    Reads API_TOKENS (comma-separated), ADMIN_TOKENS (comma-separated).
    All tokens in API_TOKENS receive 'infer' role unless also in ADMIN_TOKENS.

    Returns:
        Dict mapping raw token string to TokenInfo.

    Example:
        >>> store = _load_token_store()
    """
    store: dict[str, TokenInfo] = {}

    raw_tokens  = os.environ.get("API_TOKENS", "")
    raw_admins  = os.environ.get("ADMIN_TOKENS", "")

    admin_set = {t.strip() for t in raw_admins.split(",") if t.strip()}

    for tok in raw_tokens.split(","):
        tok = tok.strip()
        if not tok:
            continue
        role = "admin" if tok in admin_set else "infer"
        store[tok] = TokenInfo(
            token_id=tok[:8],
            role=role,
            raw=tok,
        )

    # Also add admin tokens not in API_TOKENS
    for tok in admin_set:
        if tok and tok not in store:
            store[tok] = TokenInfo(token_id=tok[:8], role="admin", raw=tok)

    return store


# Build once at import time; refreshed on each request via dependency
# (acceptable for this scale — hot-reload not needed)
_TOKEN_STORE: dict[str, TokenInfo] = {}

_security = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

async def get_current_token(
    request:     Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> TokenInfo:
    """Extract and validate a Bearer token from the Authorization header.

    Records authentication attempt in structured logs.

    Args:
        request:     FastAPI Request object (for request_id).
        credentials: HTTPBearer credentials (auto-injected).

    Returns:
        TokenInfo for the authenticated caller.

    Raises:
        HTTPException 401: Missing or invalid token.

    Example:
        >>> @router.get("/v1/health")
        ... async def health(token=Depends(get_current_token)):
        ...     return {"ok": True}
    """
    if credentials is None or not credentials.credentials:
        _logger.warning(
            "auth_failed reason=missing_token "
            "request_id=%s path=%s",
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
    info  = _TOKEN_STORE.get(token)

    if info is None:
        _logger.warning(
            "auth_failed reason=invalid_token "
            "request_id=%s path=%s",
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
        info.token_id,
        info.role,
        request.url.path,
        getattr(request.state, "request_id", "unknown"),
    )
    return info


def require_role(role: str) -> Callable:
    """Return a FastAPI dependency that enforces a minimum role.

    Role hierarchy: admin > infer.

    Args:
        role: Required role string ('infer' or 'admin').

    Returns:
        FastAPI dependency callable.

    Example:
        >>> @router.post("/v1/admin/promote")
        ... async def promote(tok=Depends(require_role("admin"))):
        ...     ...
    """
    _ROLE_ORDER = {"infer": 0, "admin": 1}

    async def _dependency(
        token_info: TokenInfo = Depends(get_current_token),
    ) -> TokenInfo:
        """Inner dependency that checks the token's role level.

        Args:
            token_info: Authenticated TokenInfo (injected by get_current_token).

        Returns:
            The same TokenInfo if role check passes.

        Raises:
            HTTPException 403: Insufficient role.

        Example:
            >>> tok = await _dependency(token_info)
        """
        required_level = _ROLE_ORDER.get(role, 0)
        actual_level   = _ROLE_ORDER.get(token_info.role, 0)
        if actual_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required; caller has '{token_info.role}'.",
            )
        return token_info

    return _dependency
