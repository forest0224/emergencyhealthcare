"""
tests/test_schemas.py
======================
Pydantic v2 schema validation tests.
"""
import pytest
from pydantic import ValidationError
from backend.schemas import PatientIn, PredictionOut, ErrorOut


def _valid() -> dict:
    return dict(age=45, chief_complaint="chest pain")


class TestPatientIn:
    def test_valid_minimal_dumps(self):
        """Valid minimal payload serialises without error."""
        p = PatientIn(**_valid())
        assert p.model_dump()["age"] == 45

    def test_rejects_age_200(self):
        """age=200 exceeds max 120 → ValidationError."""
        with pytest.raises(ValidationError) as exc:
            PatientIn(age=200, chief_complaint="x")
        assert any(e["loc"][-1] == "age" for e in exc.value.errors())

    def test_rejects_spo2_150(self):
        """spo2=150 exceeds max 100 → ValidationError."""
        with pytest.raises(ValidationError) as exc:
            PatientIn(age=40, chief_complaint="x", spo2=150)
        assert any(e["loc"][-1] == "spo2" for e in exc.value.errors())

    def test_missing_chief_complaint(self):
        """chief_complaint is required → ValidationError."""
        with pytest.raises(ValidationError):
            PatientIn(age=40)

    def test_acuity_target_not_a_field(self):
        """acuity_target is output-only — PatientIn has no such field."""
        p = PatientIn(**_valid())
        assert not hasattr(p, "acuity_target")

    def test_gender_normalised(self):
        """'male' → 'M', unknown → None."""
        assert PatientIn(age=30, chief_complaint="x", gender="male").gender == "M"
        assert PatientIn(age=30, chief_complaint="x", gender="X").gender is None

    def test_optional_vitals_default_none(self):
        """All vitals optional; absent ⇒ None."""
        p = PatientIn(**_valid())
        assert p.systolic_bp is None
        assert p.spo2 is None

    def test_arrival_hour_bounds(self):
        """arrival_hour 0 and 23 accepted; 24 rejected."""
        PatientIn(age=40, chief_complaint="x", arrival_hour=0)
        PatientIn(age=40, chief_complaint="x", arrival_hour=23)
        with pytest.raises(ValidationError):
            PatientIn(age=40, chief_complaint="x", arrival_hour=24)


class TestPredictionOut:
    def test_disclaimer_always_present(self):
        """Disclaimer field contains required clinical phrase."""
        out = PredictionOut(
            proba={"1": 0.6, "2": 0.3, "3": 0.08, "4": 0.02},
            predicted=1, confidence=0.6, model_version="v1",
        )
        assert "licensed clinician" in out.disclaimer.lower()

    def test_proba_is_dict(self):
        """proba must be a dict keyed by acuity string."""
        out = PredictionOut(
            proba={"1": 0.5, "2": 0.3, "3": 0.1, "4": 0.1},
            predicted=1, confidence=0.5, model_version="v1",
        )
        assert isinstance(out.proba, dict) and "1" in out.proba


class TestErrorOut:
    def test_round_trip(self):
        e = ErrorOut(error_code="FOO", message="bar", request_id="r1")
        assert e.model_dump()["error_code"] == "FOO"
