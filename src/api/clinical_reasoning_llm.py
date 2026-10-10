"""
src/api/clinical_reasoning_llm.py
===================================
LLM-backed clinical reasoning for the ED-Triage API.

Supports two backends:
  A. IBM watsonx.ai (ibm/granite-13b-chat-v2)
     env: WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL,
          WATSONX_MODEL_ID (default 'ibm/granite-13b-chat-v2')
     decoding: greedy, temp=0.0, max_new_tokens=700, stop=['</json>'].
  B. OpenAI-compatible
     env: OPENAI_API_KEY, OPENAI_MODEL (default 'gpt-4o-mini')

On any failure after 1 retry, raises ClinicalReasonerError.
The caller (routes.py) catches it and falls back to rule_based_rationale().
The API always returns HTTP 200.

Module-level constants SYSTEM_PROMPT and USER_PROMPT_TEMPLATE are exposed
for testing and prompt auditing.

Public API
----------
class ClinicalReasonerError(Exception)
class ClinicalReasoner
    reason(patient, ml_output, shap_top, request_id) -> dict

Usage
-----
>>> cr = ClinicalReasoner(backend="watsonx")
>>> result = cr.reason(patient_dict, ml_output, shap_top, "req-001")
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from typing import Any

from src.api.pii_scrubber import scrub_free_text

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional backends — graceful degradation
# ---------------------------------------------------------------------------
try:
    from ibm_watsonx_ai.foundation_models import ModelInference as _WxModel
    from ibm_watsonx_ai import Credentials as _WxCredentials
    _HAS_WATSONX = True
except ImportError:
    _HAS_WATSONX = False

try:
    from openai import OpenAI as _OpenAIClient
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# ---------------------------------------------------------------------------
# Custom exception — raised after exhausting retries
# ---------------------------------------------------------------------------

class ClinicalReasonerError(Exception):
    """Raised when the LLM backend fails after all retries.

    Callers should catch this and invoke rule_based_rationale() as fallback.

    Example:
        >>> try:
        ...     result = cr.reason(patient, ml_output, shap_top, "req-1")
        ... except ClinicalReasonerError:
        ...     result = rule_based_rationale(patient, ml_output)
    """


# ---------------------------------------------------------------------------
# Module-level prompt constants (exposed for testing / auditing)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a clinical decision-support assistant for an emergency department "
    "triage system. You receive ANONYMISED patient vitals — NO names, MRNs, "
    "dates, or any PHI — and an ML model prediction.\n\n"
    "Output ONLY valid JSON exactly matching this schema (no markdown fences, "
    "no extra keys, no commentary):\n"
    "{\n"
    '  "differential_risks": ["<condition>", ...],\n'
    '  "abnormal_vitals": ["<vital description>", ...],\n'
    '  "immediate_actions": ["<action>", ...],\n'
    '  "red_flags": ["<critical finding>", ...],\n'
    '  "safety_note": "<1-2 sentence safety annotation>"\n'
    "}</json>\n\n"
    "Rules:\n"
    "• Base reasoning on NEWS2 and qSOFA clinical guidelines.\n"
    "• Do NOT invent information not present in the input.\n"
    "• If a vital is N/A, do not speculate about it.\n"
    "• Keep immediate_actions ordered by clinical priority."
)

USER_PROMPT_TEMPLATE: str = """\
Patient vitals (fully anonymised):
  Age: {age}
  Chief complaint: {chief_complaint}
  Heart rate: {heart_rate} bpm
  Systolic BP: {systolic_bp} mmHg
  Diastolic BP: {diastolic_bp} mmHg
  Respiratory rate: {respiratory_rate} /min
  SpO2: {spo2}%
  Temperature: {temperature} °C
  Pain score: {pain_score}/10
  Arrival hour: {arrival_hour}

ML model output:
  Predicted acuity: {predicted}  (1=Critical … 4=Non-Urgent)
  Confidence: {confidence:.0%}
  P(acuity=1): {p1:.3f}  P(acuity=2): {p2:.3f}
  P(acuity=3): {p3:.3f}  P(acuity=4): {p4:.3f}

Top SHAP drivers: {shap_summary}

Output JSON only (end with </json>):"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_user_prompt(patient: dict, ml_output: dict, shap_top: Any) -> str:
    """Format the user-facing LLM prompt with PII-scrubbed patient data.

    Args:
        patient:   Patient dict (vitals only — no names/MRNs).
        ml_output: Output dict from TriageModel.predict_proba().
        shap_top:  SHAP top-features dict or list.

    Returns:
        Formatted user prompt string ready to send to the LLM.

    Example:
        >>> prompt = _format_user_prompt(patient, ml_out, shap_dict)
    """
    def _fmt(val):
        if val is None or (isinstance(val, float) and val != val):
            return "N/A"
        return str(val)

    # SHAP summary — handle both dict-of-lists and flat list formats
    if isinstance(shap_top, dict):
        pred_cls = str(ml_output.get("predicted", "1"))
        entries  = shap_top.get(pred_cls, [])
    elif isinstance(shap_top, list):
        entries = shap_top
    else:
        entries = []

    shap_str = ", ".join(
        f"{e.get('feature', '?')} ({e.get('shap_value', 0):+.3f})"
        for e in entries[:5]
    ) or "N/A"

    # Proba — handle dict {"1":0.6,...} or list [0.6,...]
    proba = ml_output.get("proba", {})
    if isinstance(proba, dict):
        p1 = float(proba.get("1", 0))
        p2 = float(proba.get("2", 0))
        p3 = float(proba.get("3", 0))
        p4 = float(proba.get("4", 0))
    elif isinstance(proba, list) and len(proba) >= 4:
        p1, p2, p3, p4 = float(proba[0]), float(proba[1]), float(proba[2]), float(proba[3])
    else:
        p1 = p2 = p3 = p4 = 0.0

    # Scrub chief complaint — the only free-text field
    cc = scrub_free_text(str(patient.get("chief_complaint", "")))

    return USER_PROMPT_TEMPLATE.format(
        age              =_fmt(patient.get("age")),
        chief_complaint  =cc,
        heart_rate       =_fmt(patient.get("heart_rate")),
        systolic_bp      =_fmt(patient.get("systolic_bp")),
        diastolic_bp     =_fmt(patient.get("diastolic_bp")),
        respiratory_rate =_fmt(patient.get("respiratory_rate")),
        spo2             =_fmt(patient.get("spo2")),
        temperature      =_fmt(patient.get("temperature")),
        pain_score       =_fmt(patient.get("pain_score")),
        arrival_hour     =_fmt(patient.get("arrival_hour")),
        predicted        =ml_output.get("predicted", "?"),
        confidence       =float(ml_output.get("confidence", 0.0)),
        p1=p1, p2=p2, p3=p3, p4=p4,
        shap_summary     =shap_str,
    )


def _parse_llm_json(raw: str) -> dict:
    """Extract and validate JSON from raw LLM output.

    Strips markdown fences and the </json> stop token if present.
    Validates that all required keys are present.

    Args:
        raw: Raw LLM string response.

    Returns:
        Parsed dict with required keys:
        differential_risks, abnormal_vitals, immediate_actions,
        red_flags, safety_note.

    Raises:
        ValueError: If JSON is missing or required keys absent.
        json.JSONDecodeError: If JSON cannot be parsed.

    Example:
        >>> d = _parse_llm_json('{"differential_risks": [], ...}</json>')
    """
    text = raw.strip()
    # Remove stop token
    text = text.replace("</json>", "").strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(l for l in lines if not l.startswith("```"))

    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM response (len={len(raw)}).")

    parsed = json.loads(text[start:end])

    required = {"differential_risks", "abnormal_vitals",
                "immediate_actions", "red_flags", "safety_note"}
    missing  = required - set(parsed.keys())
    if missing:
        raise ValueError(f"LLM JSON missing required keys: {sorted(missing)}")

    return parsed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ClinicalReasoner:
    """LLM-backed clinical reasoning module.

    Supports IBM watsonx.ai and OpenAI-compatible backends.
    On any backend failure after 1 retry, raises ClinicalReasonerError.
    The caller is responsible for falling back to rule_based_rationale().

    Args:
        backend: 'watsonx' (default) or 'openai'.

    Example:
        >>> cr = ClinicalReasoner(backend="watsonx")
        >>> try:
        ...     result = cr.reason(patient, ml_output, shap_top, "req-001")
        ... except ClinicalReasonerError:
        ...     result = rule_based_rationale(patient, ml_output)
    """

    def __init__(self, backend: str = "watsonx") -> None:
        """Initialise the reasoner with the specified backend.

        Args:
            backend: 'watsonx' or 'openai'.

        Example:
            >>> cr = ClinicalReasoner("openai")
        """
        self._backend      = backend.lower()
        self._client: Any  = None
        self._wx_model_id  = os.environ.get("WATSONX_MODEL_ID", "ibm/granite-13b-chat-v2")
        self._openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        if self._backend == "watsonx":
            if not _HAS_WATSONX:
                warnings.warn(
                    "[warn] ibm-watsonx-ai not installed; "
                    "ClinicalReasoner will raise ClinicalReasonerError on every call.",
                    stacklevel=2,
                )
            else:
                self._init_watsonx()

        elif self._backend == "openai":
            if not _HAS_OPENAI:
                warnings.warn(
                    "[warn] openai not installed; "
                    "ClinicalReasoner will raise ClinicalReasonerError on every call.",
                    stacklevel=2,
                )
            else:
                self._init_openai()
        else:
            warnings.warn(
                f"[warn] Unknown LLM backend '{backend}'; "
                "ClinicalReasoner will raise on every call.",
                stacklevel=2,
            )

    def _init_watsonx(self) -> None:
        """Initialise the watsonx.ai model client.

        Example:
            >>> cr._init_watsonx()
        """
        try:
            creds = _WxCredentials(
                api_key=os.environ["WATSONX_API_KEY"],
                url=os.environ.get(
                    "WATSONX_URL", "https://us-south.ml.cloud.ibm.com"
                ),
            )
            self._client = _WxModel(
                model_id   =self._wx_model_id,
                credentials=creds,
                project_id =os.environ["WATSONX_PROJECT_ID"],
            )
        except KeyError as exc:
            warnings.warn(
                f"[warn] watsonx env var missing ({exc}); "
                "ClinicalReasoner will raise on every call.",
                stacklevel=2,
            )
            self._client = None

    def _init_openai(self) -> None:
        """Initialise the OpenAI client.

        Example:
            >>> cr._init_openai()
        """
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            warnings.warn(
                "[warn] OPENAI_API_KEY not set; "
                "ClinicalReasoner will raise on every call.",
                stacklevel=2,
            )
            self._client = None
        else:
            self._client = _OpenAIClient(api_key=api_key)

    # ------------------------------------------------------------------ #

    def _call_watsonx(self, user_prompt: str) -> str:
        """Call watsonx.ai and return raw text.

        Args:
            user_prompt: Formatted user prompt (already PII-scrubbed).

        Returns:
            Raw LLM text response.

        Raises:
            Exception: On network / API error.

        Example:
            >>> raw = cr._call_watsonx(prompt)
        """
        params = {
            "max_new_tokens":  700,
            "temperature":     0.0,
            "decoding_method": "greedy",
            "stop_sequences":  ["</json>"],
        }
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        response = self._client.generate_text(
            prompt=full_prompt,
            params=params,
        )
        return str(response)

    def _call_openai(self, user_prompt: str) -> str:
        """Call OpenAI-compatible API and return raw text.

        Args:
            user_prompt: Formatted user prompt (already PII-scrubbed).

        Returns:
            Raw LLM text response.

        Raises:
            Exception: On network / API error.

        Example:
            >>> raw = cr._call_openai(prompt)
        """
        response = self._client.chat.completions.create(
            model=self._openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=700,
            stop=["</json>"],
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------ #

    def reason(
        self,
        patient:    dict,
        ml_output:  dict,
        shap_top:   Any,
        request_id: str,
    ) -> dict:
        """Generate clinical rationale for a triage prediction via LLM.

        Calls the configured backend.  On JSON schema failure retries once,
        then raises ClinicalReasonerError.  On hard errors (network, auth)
        raises immediately without retry.

        The LLM NEVER sees raw PHI: chief_complaint is PII-scrubbed inside
        _format_user_prompt() before being sent.

        Args:
            patient:    Patient dict (vitals; PII scrubbed internally).
            ml_output:  Output from TriageModel.predict_proba().
            shap_top:   Top SHAP dict or list.
            request_id: UUID for audit trail correlation.

        Returns:
            Dict with keys matching RationaleOut LLM fields:
                differential_risks, abnormal_vitals, immediate_actions,
                red_flags, safety_note, llm_backend.

        Raises:
            ClinicalReasonerError: After exhausting retries or if no
                client is configured.

        Example:
            >>> result = cr.reason(patient, ml_output, shap_top, "req-001")
            >>> print(result["differential_risks"])
        """
        if self._client is None:
            raise ClinicalReasonerError(
                f"No {self._backend} client configured (check env vars or package install)."
            )

        user_prompt = _format_user_prompt(patient, ml_output, shap_top)
        call_fn     = self._call_watsonx if self._backend == "watsonx" else self._call_openai

        last_exc: Exception | None = None
        for attempt in range(1, 3):  # attempts 1, 2
            try:
                raw    = call_fn(user_prompt)
                parsed = _parse_llm_json(raw)
                parsed["llm_backend"] = self._backend
                _logger.info(
                    "llm_ok request_id=%s backend=%s attempt=%d",
                    request_id, self._backend, attempt,
                )
                return parsed

            except (json.JSONDecodeError, ValueError) as exc:
                last_exc = exc
                _logger.warning(
                    "llm_schema_error request_id=%s attempt=%d error=%s",
                    request_id, attempt, exc,
                )
                # Retry on schema / parse failures
                continue

            except Exception as exc:
                # Hard error (network, auth, quota) — raise immediately
                _logger.error(
                    "llm_hard_error request_id=%s attempt=%d error_type=%s",
                    request_id, attempt, type(exc).__name__,
                )
                raise ClinicalReasonerError(
                    f"LLM hard error on attempt {attempt}: {type(exc).__name__}"
                ) from exc

        raise ClinicalReasonerError(
            f"LLM returned invalid JSON after 2 attempts. Last error: {last_exc}"
        )
