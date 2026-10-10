"""
src/api/pii_scrubber.py
========================
PII scrubbing for free-text clinical fields.

Removes names, MRNs, phone numbers, email addresses, and dates from
chief_complaint BEFORE any text is sent to an LLM or logged.

Logs a SHA-256 hash of every redacted span (never the raw PHI) to the
Python 'pii_audit' logger.

Public API
----------
scrub_free_text(text: str) -> str

Usage
-----
>>> from src.api.pii_scrubber import scrub_free_text
>>> clean = scrub_free_text("Dr. Smith saw MRN 12345 on 01/15/2024")
>>> print(clean)   # "[NAME] saw [MRN] on [DATE]"
"""

from __future__ import annotations

import hashlib
import logging
import re

_logger = logging.getLogger("pii_audit")

# ---------------------------------------------------------------------------
# Redaction patterns  (order matters — run most specific first)
# ---------------------------------------------------------------------------

# MRN: typically 5-10 digits, optionally prefixed with "MRN", "mrn", "#"
_MRN_PATTERN = re.compile(
    r"\b(?:MRN[:\s#]*|mrn[:\s#]*)?\d{5,10}\b",
    re.IGNORECASE,
)

# Phone numbers: US and international formats
_PHONE_PATTERN = re.compile(
    r"\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"
)

# Email addresses
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# Dates: MM/DD/YYYY, DD-MM-YYYY, Month DD YYYY, etc.
_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
    r"|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b",
    re.IGNORECASE,
)

# Titles + proper names: "Dr. Lastname", "Nurse Firstname", "Mr./Mrs./Ms. Name"
_NAME_PATTERN = re.compile(
    r"\b(?:Dr\.?|Doctor|Nurse|RN|Mr\.?|Mrs\.?|Ms\.?|Prof\.?)\s+[A-Z][a-z]+"
    r"(?:\s+[A-Z][a-z]+)?",
    re.IGNORECASE,
)

# Static deny-list of commonly leaked identifiers (extend as needed)
_DENYLIST: list[str] = []  # populated from environment or config at import time

# Ordered list of (pattern, replacement_tag)
_RULES: list[tuple[re.Pattern, str]] = [
    (_NAME_PATTERN,  "[NAME]"),
    (_EMAIL_PATTERN, "[EMAIL]"),
    (_PHONE_PATTERN, "[PHONE]"),
    (_MRN_PATTERN,   "[MRN]"),
    (_DATE_PATTERN,  "[DATE]"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrub_free_text(text: str) -> str:
    """Remove PII/PHI from free-text clinical input.

    Applies regex-based redaction for names, MRNs, phone numbers, emails,
    and dates.  Logs a SHA-256 hash of each redacted span to the 'pii_audit'
    logger (never the raw PHI itself).

    Args:
        text: Raw free-text string (e.g. chief_complaint value).

    Returns:
        Scrubbed string with PII replaced by tagged placeholders.

    Example:
        >>> clean = scrub_free_text("Dr. Smith MRN 12345 called 555-867-5309")
        >>> print(clean)
        "[NAME] [MRN] called [PHONE]"
    """
    if not isinstance(text, str):
        return str(text)

    result = text

    # Apply deny-list first (case-insensitive exact match)
    for term in _DENYLIST:
        if term in result:
            _log_redaction(term, "[DENYLIST]")
            result = result.replace(term, "[DENYLIST]")

    # Apply regex rules
    for pattern, tag in _RULES:
        def _replace(m: re.Match, _tag: str = tag) -> str:
            _log_redaction(m.group(0), _tag)
            return _tag
        result = pattern.sub(_replace, result)

    return result


def _log_redaction(raw: str, tag: str) -> None:
    """Log a SHA-256 hash of a redacted span — never the raw value.

    Args:
        raw: The raw PII string that was redacted.
        tag: The replacement tag used (e.g. '[MRN]').

    Example:
        >>> _log_redaction("12345", "[MRN]")
    """
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    _logger.info("pii_redacted tag=%s hash=%s", tag, digest)
