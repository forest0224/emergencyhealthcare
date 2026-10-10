"""
src/api/fallback_rationale.py
==============================
Pure-Python rule-based clinical rationale generator.

Used whenever the LLM is unreachable, times out, or returns malformed JSON.
Ensures the ED is NEVER blocked by an LLM outage.

Based on NEWS2 and qSOFA thresholds from:
  Royal College of Physicians. National Early Warning Score (NEWS) 2. 2017.

Public API
----------
rule_based_rationale(patient: dict, ml_output: dict) -> dict

Usage
-----
>>> from src.api.fallback_rationale import rule_based_rationale
>>> r = rule_based_rationale({"heart_rate": 130, "spo2": 88}, {"predicted": 1})
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Threshold constants (NEWS2 / qSOFA)
# ---------------------------------------------------------------------------
_NORMAL_RANGES = {
    "heart_rate":       (51,  90),
    "systolic_bp":      (111, 219),
    "diastolic_bp":     (61,  89),
    "respiratory_rate": (12,  20),
    "spo2":             (96,  100),
    "temperature":      (36.1, 38.0),
    "pain_score":       (0,   3),
}

_CRITICAL_THRESHOLDS = {
    "heart_rate":       {"low": 40,  "high": 131},
    "systolic_bp":      {"low": 90,  "high": 220},
    "respiratory_rate": {"low": 8,   "high": 25},
    "spo2":             {"low": 91,  "high": None},
    "temperature":      {"low": 35.0, "high": 39.1},
}

_ACUITY_LABELS = {1: "CRITICAL", 2: "EMERGENT", 3: "URGENT", 4: "NON-URGENT"}

_ACUITY_ACTIONS = {
    1: [
        "Immediate physician assessment required.",
        "Activate resuscitation resources (code team if needed).",
        "Continuous cardiac and SpO2 monitoring.",
        "IV access × 2; draw stat labs (ABG, CBC, BMP, troponin).",
        "Consider immediate airway management.",
    ],
    2: [
        "Physician assessment within 15 minutes.",
        "Cardiac monitoring and IV access.",
        "Stat ECG if chest pain or palpitations.",
        "Draw relevant labs and imaging as indicated.",
        "Reassess vitals every 15 minutes.",
    ],
    3: [
        "Nurse assessment within 30 minutes.",
        "Vital signs monitoring every 30 minutes.",
        "Initiate appropriate diagnostic workup.",
        "Physician assessment within 1 hour.",
    ],
    4: [
        "Routine assessment by nursing staff.",
        "Vital signs on arrival; repeat if condition changes.",
        "Physician assessment within 2 hours.",
    ],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _identify_abnormal_vitals(patient: dict) -> list[str]:
    """Identify vitals outside normal ranges.

    Args:
        patient: Patient dict with vital-sign keys.

    Returns:
        List of human-readable abnormal vital descriptions.

    Example:
        >>> abnormal = _identify_abnormal_vitals({"heart_rate": 135, "spo2": 88})
        >>> print(abnormal)
        ['Heart rate 135 bpm (normal 51-90)', 'SpO2 88% (critically low)']
    """
    findings: list[str] = []

    hr  = patient.get("heart_rate")
    sbp = patient.get("systolic_bp")
    dbp = patient.get("diastolic_bp")
    rr  = patient.get("respiratory_rate")
    spo2 = patient.get("spo2")
    temp = patient.get("temperature")
    pain = patient.get("pain_score")

    if hr is not None:
        if hr < 40:
            findings.append(f"Heart rate {hr:.0f} bpm — critical bradycardia (normal 51-90)")
        elif hr > 130:
            findings.append(f"Heart rate {hr:.0f} bpm — critical tachycardia (normal 51-90)")
        elif hr > 90:
            findings.append(f"Heart rate {hr:.0f} bpm — elevated (normal 51-90)")
        elif hr < 51:
            findings.append(f"Heart rate {hr:.0f} bpm — low (normal 51-90)")

    if sbp is not None:
        if sbp < 90:
            findings.append(f"Systolic BP {sbp:.0f} mmHg — hypotension (normal ≥111)")
        elif sbp >= 220:
            findings.append(f"Systolic BP {sbp:.0f} mmHg — hypertensive crisis")
        elif sbp >= 180:
            findings.append(f"Systolic BP {sbp:.0f} mmHg — severe hypertension")

    if rr is not None:
        if rr <= 8:
            findings.append(f"Respiratory rate {rr:.0f} bpm — critically low (normal 12-20)")
        elif rr >= 25:
            findings.append(f"Respiratory rate {rr:.0f} bpm — critically elevated (normal 12-20)")
        elif rr >= 21:
            findings.append(f"Respiratory rate {rr:.0f} bpm — elevated (normal 12-20)")

    if spo2 is not None:
        if spo2 <= 91:
            findings.append(f"SpO2 {spo2:.0f}% — critically low (normal ≥96%)")
        elif spo2 <= 95:
            findings.append(f"SpO2 {spo2:.0f}% — low (normal ≥96%)")

    if temp is not None:
        if temp <= 35.0:
            findings.append(f"Temperature {temp:.1f}°C — hypothermia (normal 36.1-38.0)")
        elif temp >= 39.1:
            findings.append(f"Temperature {temp:.1f}°C — high fever (normal 36.1-38.0)")
        elif temp >= 38.0:
            findings.append(f"Temperature {temp:.1f}°C — fever (normal 36.1-38.0)")

    if pain is not None and pain >= 7:
        findings.append(f"Pain score {pain:.0f}/10 — severe pain")

    return findings


def _build_rationale(
    patient: dict,
    predicted: int,
    abnormal_vitals: list[str],
) -> str:
    """Build a concise clinical rationale string.

    Args:
        patient:         Patient dict.
        predicted:       Predicted acuity (1-4).
        abnormal_vitals: List of abnormal vital descriptions.

    Returns:
        Multi-sentence rationale string.

    Example:
        >>> r = _build_rationale(patient, 1, ["SpO2 88%"])
    """
    label = _ACUITY_LABELS.get(predicted, "UNKNOWN")
    cc    = str(patient.get("chief_complaint", "unspecified complaint"))
    age   = patient.get("age")
    age_str = f"{int(age)}-year-old" if age is not None else "patient"

    parts: list[str] = [
        f"The {age_str} presents with {cc}.",
    ]

    if abnormal_vitals:
        parts.append(
            "Abnormal findings: " + "; ".join(abnormal_vitals) + "."
        )
    else:
        parts.append("Vital signs are within acceptable ranges at this time.")

    # NEWS2 / qSOFA quick summary
    hr  = patient.get("heart_rate", 80)
    sbp = patient.get("systolic_bp", 120)
    rr  = patient.get("respiratory_rate", 16)
    qsofa = (
        int((hr or 0) > 100)
        + int((sbp or 120) < 100)
        + int((rr or 16) >= 22)
    )
    if qsofa >= 2:
        parts.append(
            f"qSOFA proxy score {qsofa}/3 suggests possible sepsis or "
            "haemodynamic compromise; urgent sepsis screening recommended."
        )

    parts.append(
        f"Based on vital-sign pattern and chief complaint, the ML model "
        f"assigns acuity {predicted} ({label}). "
        f"Clinical confirmation required."
    )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Differential risk map by acuity
# ---------------------------------------------------------------------------

_DIFFERENTIAL_RISKS: dict[int, list[str]] = {
    1: [
        "Acute coronary syndrome / STEMI",
        "Massive pulmonary embolism",
        "Tension pneumothorax",
        "Haemorrhagic shock",
        "Septic shock",
        "Stroke (large vessel occlusion)",
    ],
    2: [
        "Acute coronary syndrome (NSTEMI / unstable angina)",
        "Pulmonary embolism",
        "Severe sepsis",
        "Acute respiratory failure",
        "Hypertensive emergency",
        "Intracranial haemorrhage",
    ],
    3: [
        "Community-acquired pneumonia",
        "Urinary tract infection with systemic signs",
        "Moderate dehydration",
        "Fracture / musculoskeletal injury",
        "Moderate asthma exacerbation",
    ],
    4: [
        "Upper respiratory tract infection",
        "Minor soft-tissue injury",
        "Prescription refill / medication query",
    ],
}


def _build_red_flags(patient: dict, abnormal: list[str]) -> list[str]:
    """Identify red-flag findings requiring immediate escalation.

    Args:
        patient:  Patient dict with vital keys.
        abnormal: Already-identified list of abnormal vital descriptions.

    Returns:
        List of red-flag strings.

    Example:
        >>> flags = _build_red_flags({"systolic_bp": 70, "spo2": 85}, [])
    """
    flags: list[str] = []
    sbp  = patient.get("systolic_bp")
    spo2 = patient.get("spo2")
    hr   = patient.get("heart_rate")
    rr   = patient.get("respiratory_rate")
    temp = patient.get("temperature")

    if sbp is not None and sbp < 90:
        flags.append(f"Critical hypotension — SBP {sbp:.0f} mmHg")
    if spo2 is not None and spo2 <= 91:
        flags.append(f"Critically low SpO2 — {spo2:.0f}%")
    if hr is not None and (hr < 40 or hr > 150):
        flags.append(f"Extreme heart rate — {hr:.0f} bpm")
    if rr is not None and (rr <= 8 or rr >= 30):
        flags.append(f"Dangerously abnormal respiratory rate — {rr:.0f}/min")
    if temp is not None and temp <= 35.0:
        flags.append(f"Hypothermia — {temp:.1f}°C")
    if temp is not None and temp >= 40.0:
        flags.append(f"Hyperpyrexia — {temp:.1f}°C")

    return flags


def rule_based_rationale(patient: dict, ml_output: dict) -> dict:
    """Generate a rule-based clinical rationale without an LLM.

    Uses NEWS2/qSOFA thresholds to identify abnormal vitals,
    red flags, differential risks, and immediate actions.
    Returns the same schema as ClinicalReasoner.reason() so the API
    response is always uniform regardless of LLM availability.

    Args:
        patient:   Patient dict (keys matching CSV column names).
        ml_output: Output dict from TriageModel.predict_proba().

    Returns:
        Dict with keys matching the RationaleOut LLM contract:
            differential_risks – list[str]
            abnormal_vitals    – list[str]
            immediate_actions  – list[str]
            red_flags          – list[str]
            safety_note        – str
            llm_backend        – "rule_based"

    Example:
        >>> r = rule_based_rationale(
        ...     {"age": 70, "heart_rate": 135, "spo2": 88},
        ...     {"predicted": 1, "confidence": 0.72}
        ... )
        >>> print(r["red_flags"])
    """
    predicted = int(ml_output.get("predicted", 3))
    abnormal  = _identify_abnormal_vitals(patient)
    red_flags = _build_red_flags(patient, abnormal)
    actions   = _ACUITY_ACTIONS.get(predicted, _ACUITY_ACTIONS[3])
    diffs     = _DIFFERENTIAL_RISKS.get(predicted, [])

    confidence = float(ml_output.get("confidence", 0.0))
    label      = _ACUITY_LABELS.get(predicted, "UNKNOWN")
    imputed    = ml_output.get("imputed_fields", [])

    safety_parts: list[str] = [
        f"Rule-based rationale (LLM unavailable). "
        f"ML model assigned acuity {predicted} ({label}) "
        f"with {confidence:.0%} confidence."
    ]
    if imputed:
        safety_parts.append(
            f"Note: {len(imputed)} vital(s) were imputed "
            f"({', '.join(imputed)}); treat prediction with additional caution."
        )
    if red_flags:
        safety_parts.append("Critical findings present — immediate escalation required.")

    return {
        "differential_risks": diffs,
        "abnormal_vitals":    abnormal,
        "immediate_actions":  actions,
        "red_flags":          red_flags,
        "safety_note":        " ".join(safety_parts),
        "llm_backend":        "rule_based",
    }
