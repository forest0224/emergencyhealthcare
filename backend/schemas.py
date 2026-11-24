"""
backend/schemas.py
===================
Pydantic v2 request/response schemas for the ED-Triage API.

Models
------
PatientIn     – validated patient input (all vitals optional, range-checked)
PredictionOut – ML-only prediction response
RationaleOut  – Full triage with LLM clinical rationale (strict LLM contract)
ErrorOut      – Structured error envelope
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class PatientIn(BaseModel):
    """Validated patient triage input.

    All vitals are optional — missing values are imputed by the ML pipeline.
    Ranges are software-validation limits; they are NOT definitions of
    medically normal values.
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
        description="Free-text chief complaint (PII-scrubbed before any LLM call).",
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
        max_length=128,
        description="Optional encounter ID for audit linkage (SHA-256 hashed before storage).",
    )

    @field_validator("chief_complaint", mode="before")
    @classmethod
    def trim_complaint(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: Optional[str]) -> Optional[str]:
        """Normalise gender to 'M' | 'F' | None."""
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
    """ML-only triage prediction response (fast path)."""
    proba: dict[str, float] = Field(
        description='Probability per acuity class. Keys "1".."4", values 0–1.'
    )
    predicted: int = Field(description="Most likely acuity class (1–4).")
    confidence: float = Field(description="Probability of the predicted class.")
    imputed_fields: list[str] = Field(
        default_factory=list,
        description="Vital fields that were missing and imputed.",
    )
    shap_top: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict,
        description="Top SHAP feature contributions keyed by acuity class.",
    )
    safety_flags: list[str] = Field(
        default_factory=list,
        description="Clinical safety alerts.",
    )
    model_config = ConfigDict(protected_namespaces=())
    latency_ms: int | None = None
    model_version: str = Field(description="Active model version ID.")
    disclaimer: str = Field(
        default=(
            "Decision-support only. Not a diagnosis. "
            "Confirm with a licensed clinician."
        )
    )


class RationaleOut(BaseModel):
    """Full triage output including LLM-generated or rule-based rationale."""
    # ML output fields
    predicted_acuity: int = Field(description="Predicted acuity class (1–4).")
    confidence: float = Field(description="Probability of predicted class.")
    proba: dict[str, float] = Field(
        description='Probability per acuity class. Keys "1".."4".'
    )
    imputed_fields: list[str] = Field(default_factory=list)
    shap_top: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    safety_flags: list[str] = Field(default_factory=list)
    model_config = ConfigDict(protected_namespaces=())
    latency_ms: int | None = None
    model_version: str = Field(description="Active model version ID.")

    rationale: str = ""
    reasoning_model: str | None = None

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
        description="Backend used: 'openai' or 'rule_based'.",
    )
    disclaimer: str = Field(
        default=(
            "Decision-support only. Not a diagnosis. "
            "Confirm with a licensed clinician."
        )
    )


class ErrorOut(BaseModel):
    """Structured error response — never leaks PHI or stack traces."""
    error_code: str = Field(description="Machine-readable error code.")
    message: str = Field(description="Human-readable error description.")
    request_id: str = Field(description="UUID of the failing request.")
