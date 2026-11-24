from random import Random

from backend.schemas import PatientIn
from synthetic_patients import CATEGORIES, SCENARIOS, VITAL_FIELDS, DemoPatientGenerator


def test_generator_validates_1000_profiles_and_cycles_all_templates():
    generator = DemoPatientGenerator(Random(2026))
    encounter_ids = set()
    last_complaint = None
    last_vitals = {}
    first_cycle = set()

    for index in range(1_000):
        profile = generator.generate()
        validated = PatientIn.model_validate(profile)
        assert validated.systolic_bp > validated.diastolic_bp
        assert profile["encounter_id"] not in encounter_ids
        encounter_ids.add(profile["encounter_id"])
        complaint = profile["chief_complaint"]
        assert complaint != last_complaint
        last_complaint = complaint
        if index < len(SCENARIOS):
            first_cycle.add(complaint)
        vitals = tuple(profile[field] for field in VITAL_FIELDS)
        assert vitals != last_vitals.get(complaint)
        last_vitals[complaint] = vitals

    assert len(SCENARIOS) >= 30
    assert first_cycle == {scenario.complaint for scenario in SCENARIOS}


def test_category_decks_cover_all_four_groups():
    generator = DemoPatientGenerator(Random(42))
    for category in CATEGORIES:
        complaints = {s.complaint for s in SCENARIOS if s.category == category}
        assert len(complaints) > 1
        generated = {generator.generate(category)["chief_complaint"] for _ in complaints}
        assert generated == complaints
