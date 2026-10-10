"""
src/api/schemas.py
===================
Pydantic v2 request/response schemas for the ED-Triage API.

Models
------
PatientIn     – validated patient input (all vitals optional, range-checked)
PredictionOut – ML-only prediction response
RationaleOut  – Full triage with LLM clinical rationale (strict LLM contract)
ErrorOut      – Structured error envelope

Usage
-----
>>> from src.api.schemas import PatientIn, PredictionOut
>>> p = PatientIn(age=45, heart_rate=110, systolic_bp=90,
...               chief_complaint="chest pain")
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class PatientIn(BaseModel):
    """Validated patient triage input.

    All vitals are optional — missing values are imputed by the ML pipeline.
    Ranges are clinically validated; out-of-range values are rejected (422).

    Example:
        >>> p = PatientIn(age=65, heart_rate=120, systolic_bp=80,
        ...               spo2=94, chief_complaint="shortness of breath")
    """

    # Demographics
    age: float = Field(
        ...,
        ge=0, le=120,
        description="Patient age in years (0–120).",
    )
    gender: Optional[str] = Field(
        None,
        description="Biological sex: 'M', 'F', or null.",
    )
    chief_complaint: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Free-text chief complaint (will be PII-scrubbed before any LLM call).",
    )

    # Timing
    arrival_hour: Optional[int] = Field(
        None,
        ge=0, le=23,
        description="Hour of arrival (0–23, 24h clock).",
    )

    # Vitals — all Optional with clinically-validated ranges
    systolic_bp: Optional[float] = Field(
        None, ge=40, le=300,
        description="Systolic blood pressure (mmHg, 40–300).",
    )
    diastolic_bp: Optional[float] = Field(
        None, ge=20, le=200,
        description="Diastolic blood pressure (mmHg, 20–200).",
    )
    heart_rate: Optional[float] = Field(
        None, ge=20, le=300,
        description="Heart rate (bpm, 20–300).",
    )
    respiratory_rate: Optional[float] = Field(
        None, ge=4, le=60,
        description="Respiratory rate (breaths/min, 4–60).",
    )
    temperature: Optional[float] = Field(
        None, ge=30.0, le=43.0,
        description="Body temperature (°C, 30–43).",
    )
    spo2: Optional[float] = Field(
        None, ge=50, le=100,
        description="Peripheral oxygen saturation (%, 50–100).",
    )
    pain_score: Optional[float] = Field(
        None, ge=0, le=10,
        description="Self-reported pain score (0–10).",
    )

    # Optional encounter linkage — hashed before any storage
    encounter_id: Optional[str] = Field(
        None,
        description="Optional encounter ID for audit linkage (SHA-256 hashed before storage).",
    )

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: Optional[str]) -> Optional[str]:
        """Normalise gender to 'M' | 'F' | None.

        Args:
            v: Raw gender string.

        Returns:
            'M', 'F', or None for anything unrecognised.

        Example:
            >>> PatientIn.validate_gender("male")
            'M'
        """
        if v is None:
            return None
        norm = v.strip().upper()
        if norm in ("M", "MALE"):
            return "M"
        if norm in ("F", "FEMALE"):
            return "F"
        return None

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 65,
                "gender": "M",
                "chief_complaint": "chest pain and shortness of breath",
                "arrival_hour": 14,
                "systolic_bp": 80,
                "diastolic_bp": 50,
                "heart_rate": 120,
                "respiratory_rate": 24,
                "temperature": 37.2,
                "spo2": 94,
                "pain_score": 7,
            }
        }
    }


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class PredictionOut(BaseModel):
    """ML-only triage prediction response (fast path).

    ``proba`` is a dict keyed by acuity string ("1".."4") for easy JSON
    consumption.  ``shap_top`` is a dict keyed by acuity class.

    Example:
        >>> out = PredictionOut(
        ...     proba={"1": 0.6, "2": 0.3, "3": 0.08, "4": 0.02},
        ...     predicted=1, confidence=0.60,
        ...     imputed_fields=["spo2"],
        ...     shap_top={"1": [{"feature": "shock_index", "shap_value": 0.42}]},
        ...     safety_flags=["critical_acuity_immediate_review_required"],
        ...     model_version="v1",
        ... )
    """
    proba: dict[str, float] = Field(
        description=(
            'Probability per acuity class. Keys "1".."4", values 0–1. '
            "Sum ≈ 1.0."
        )
    )
    predicted:      int   = Field(description="Most likely acuity class (1–4).")
    confidence:     float = Field(description="Probability of the predicted class.")
    imputed_fields: list[str] = Field(
        default_factory=list,
        description="Vital fields that were missing and imputed by MICE.",
    )
    shap_top: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict,
        description=(
            "Top SHAP feature contributions keyed by acuity class. "
            'Each value is a list of {"feature": str, "shap_value": float}.'
        ),
    )
    safety_flags: list[str] = Field(
        default_factory=list,
        description="Clinical safety alerts (e.g. 'critical_miss_risk').",
    )
    model_version: str = Field(description="Active model version ID.")
    disclaimer: str = Field(
        default=(
            "Decision-support only. Not a diagnosis. "
            "Confirm with a licensed clinician."
        )
    )


class RationaleOut(BaseModel):
    """Full triage output including LLM-generated or rule-based clinical rationale.

    This is the strict schema that both the LLM JSON contract and the
    rule_based_rationale() fallback must satisfy.

    Example:
        >>> r = RationaleOut(
        ...     predicted_acuity=1,
        ...     confidence=0.72,
        ...     proba={"1": 0.72, "2": 0.20, "3": 0.06, "4": 0.02},
        ...     imputed_fields=[],
        ...     shap_top={"1": []},
        ...     safety_flags=[],
        ...     model_version="v1",
        ...     differential_risks=["Acute MI", "Tension pneumothorax"],
        ...     abnormal_vitals=["SpO2 93% — low"],
        ...     immediate_actions=["Immediate physician assessment"],
        ...     red_flags=["Critical hypotension (SBP 80 mmHg)"],
        ...     safety_note="High acuity probability despite any imputed vitals.",
        ...     llm_backend="watsonx",
        ... )
    """
    # ML output fields
    predicted_acuity: int   = Field(description="Predicted acuity class (1–4).")
    confidence:       float = Field(description="Probability of predicted class.")
    proba: dict[str, float] = Field(
        description='Probability per acuity class. Keys "1".."4".'
    )
    imputed_fields: list[str] = Field(default_factory=list)
    shap_top: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    safety_flags: list[str] = Field(default_factory=list)
    model_version: str = Field(description="Active model version ID.")

    # LLM / rule-based rationale fields
    differential_risks: list[str] = Field(
        default_factory=list,
        description="Differential diagnoses or risk conditions to consider.",
    )
    abnormal_vitals: list[str] = Field(
        default_factory=list,
        description="Vital signs outside normal range with descriptions.",
    )
    immediate_actions: list[str] = Field(
        default_factory=list,
        description="Ordered list of immediate clinical actions.",
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description="Critical findings that demand immediate escalation.",
    )
    safety_note: str = Field(
        default="",
        description="Brief safety annotation from the LLM or rule engine.",
    )
    llm_backend: Optional[str] = Field(
        None,
        description="Backend used: 'watsonx', 'openai', or 'rule_based'.",
    )
    disclaimer: str = Field(
        default=(
            "Decision-support only. Not a diagnosis. "
            "Confirm with a licensed clinician."
        )
    )


class ErrorOut(BaseModel):
    """Structured error response — never leaks PHI or stack traces to the client.

    Example:
        >>> e = ErrorOut(
        ...     error_code="VALIDATION_ERROR",
        ...     message="age must be <= 120",
        ...     request_id="req-abc123",
        ... )
    """
    error_code: str = Field(description="Machine-readable error code.")
    message:    str = Field(description="Human-readable error description.")
    request_id: str = Field(description="UUID of the failing request.")
