"""Scenario-based fixtures for the demo form. Ranges are illustrative test data."""

from random import Random
from typing import NamedTuple
from uuid import uuid4


class Scenario(NamedTuple):
    complaint: str
    category: str
    age: tuple[int, int]
    systolic: tuple[int, int]
    diastolic: tuple[int, int]
    pulse: tuple[int, int]
    breathing: tuple[int, int]
    temperature_tenths: tuple[int, int]
    oxygen: tuple[int, int]
    pain: tuple[int, int]


SCENARIOS = (
    # Complaint, category, age, systolic, diastolic, pulse, breathing, temperature x10, SpO2, pain.
    Scenario("Crushing chest pain with collapse", "Critical", (46, 88), (65, 88), (36, 57), (108, 148), (25, 37), (361, 374), (84, 92), (8, 10)),
    Scenario("Severe breathing difficulty at rest", "Critical", (22, 82), (92, 138), (52, 83), (113, 150), (32, 46), (362, 378), (72, 84), (0, 4)),
    Scenario("Severe allergic reaction with facial swelling", "Critical", (16, 72), (62, 86), (34, 54), (119, 155), (27, 41), (359, 373), (78, 90), (0, 4)),
    Scenario("Road accident with heavy bleeding", "Critical", (19, 72), (58, 85), (30, 52), (122, 158), (27, 40), (350, 368), (88, 97), (7, 10)),
    Scenario("Unresponsiveness after head injury", "Critical", (18, 81), (105, 166), (62, 103), (46, 82), (6, 12), (357, 375), (80, 92), (0, 4)),
    Scenario("High fever with marked confusion", "Critical", (51, 96), (62, 89), (34, 58), (113, 148), (26, 38), (391, 408), (88, 95), (1, 6)),
    Scenario("Near-drowning with persistent breathing difficulty", "Critical", (14, 67), (73, 106), (42, 65), (108, 150), (28, 42), (340, 362), (77, 89), (0, 4)),
    Scenario("Sudden abdominal pain with collapse", "Critical", (37, 90), (63, 89), (34, 56), (112, 151), (24, 37), (354, 373), (88, 98), (8, 10)),
    Scenario("Chest pressure with sweating", "Emergent", (40, 89), (99, 160), (57, 99), (83, 122), (18, 29), (361, 376), (93, 99), (6, 9)),
    Scenario("Asthma symptoms with audible wheezing", "Emergent", (12, 58), (105, 145), (59, 92), (104, 138), (24, 34), (362, 378), (88, 94), (0, 3)),
    Scenario("Sudden severe headache with nausea", "Emergent", (17, 74), (125, 185), (76, 112), (65, 107), (15, 24), (361, 378), (95, 100), (8, 10)),
    Scenario("Palpitations with lightheadedness", "Emergent", (18, 86), (95, 141), (55, 90), (131, 165), (18, 28), (362, 377), (93, 100), (0, 4)),
    Scenario("Forearm fracture with visible deformity", "Emergent", (14, 89), (108, 158), (62, 97), (88, 124), (16, 25), (361, 376), (95, 100), (7, 10)),
    Scenario("Head injury with repeated vomiting", "Emergent", (15, 85), (104, 157), (60, 98), (69, 110), (15, 24), (360, 376), (94, 100), (5, 9)),
    Scenario("Persistent vomiting with severe dehydration", "Emergent", (16, 83), (89, 111), (48, 71), (107, 133), (18, 27), (364, 385), (95, 100), (3, 7)),
    Scenario("High fever with shaking chills", "Emergent", (15, 91), (97, 142), (52, 87), (104, 132), (20, 29), (392, 405), (93, 99), (3, 7)),
    Scenario("Right-sided abdominal pain", "Urgent", (13, 73), (102, 143), (58, 91), (82, 117), (15, 24), (368, 388), (95, 100), (4, 8)),
    Scenario("Cough and fever with fatigue", "Urgent", (12, 83), (103, 148), (58, 91), (82, 114), (17, 25), (377, 394), (94, 99), (1, 5)),
    Scenario("Dizziness when standing after poor fluid intake", "Urgent", (15, 81), (92, 118), (53, 77), (83, 111), (14, 22), (361, 378), (96, 100), (0, 3)),
    Scenario("Vomiting and diarrhoea since yesterday", "Urgent", (12, 77), (98, 132), (55, 81), (83, 116), (14, 22), (365, 388), (96, 100), (2, 6)),
    Scenario("Ankle injury with inability to bear weight", "Urgent", (12, 74), (105, 142), (58, 90), (74, 110), (14, 22), (361, 376), (96, 100), (5, 8)),
    Scenario("Painful minor hot-water burn to forearm", "Urgent", (12, 76), (104, 149), (59, 93), (76, 112), (14, 23), (361, 377), (96, 100), (4, 8)),
    Scenario("Lower back pain after lifting", "Urgent", (19, 76), (105, 151), (60, 95), (71, 106), (13, 22), (360, 377), (96, 100), (4, 8)),
    Scenario("Painful urination with low-grade fever", "Urgent", (16, 83), (101, 146), (57, 92), (79, 111), (14, 23), (371, 386), (96, 100), (2, 6)),
    Scenario("Minor cut on finger", "Less Urgent", (12, 80), (99, 139), (55, 88), (64, 96), (12, 20), (360, 375), (97, 100), (1, 4)),
    Scenario("Sore throat without breathing difficulty", "Less Urgent", (12, 72), (97, 139), (54, 88), (67, 102), (12, 21), (363, 382), (96, 100), (2, 5)),
    Scenario("Mild ankle soreness after twisting", "Less Urgent", (13, 72), (100, 143), (55, 90), (62, 94), (12, 20), (360, 375), (97, 100), (1, 4)),
    Scenario("Small superficial kitchen burn", "Less Urgent", (15, 78), (101, 142), (56, 91), (64, 98), (12, 20), (360, 376), (97, 100), (1, 4)),
    Scenario("Itchy localised rash without swelling", "Less Urgent", (12, 83), (98, 141), (54, 89), (63, 97), (12, 20), (360, 376), (97, 100), (0, 2)),
    Scenario("Mild earache without discharge", "Less Urgent", (12, 71), (97, 138), (53, 85), (65, 101), (12, 21), (362, 379), (97, 100), (1, 4)),
    Scenario("Nasal congestion and mild cough", "Less Urgent", (12, 84), (98, 142), (54, 90), (64, 101), (12, 21), (361, 379), (96, 100), (0, 2)),
    Scenario("Shoulder soreness after exercise", "Less Urgent", (14, 69), (101, 144), (56, 91), (61, 97), (12, 20), (360, 376), (97, 100), (1, 4)),
)

CATEGORIES = ("Critical", "Emergent", "Urgent", "Less Urgent")
VITAL_FIELDS = ("systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate",
                "temperature", "spo2", "pain_score")


class DemoPatientGenerator:
    """Shuffle decks and avoid immediate repeats within one dashboard session."""

    def __init__(self, rng: Random | None = None):
        self.rng = rng if rng is not None else Random()
        self.decks: dict[str | None, list[Scenario]] = {}
        self.last_complaint = None
        self.last_vitals: dict[str, tuple] = {}

    def generate(self, category: str | None = None) -> dict:
        if category is not None and category not in CATEGORIES:
            raise ValueError("Unknown demo scenario category")
        deck = self.decks.setdefault(category, [])
        if not deck:
            deck.extend(s for s in SCENARIOS if category is None or s.category == category)
            self.rng.shuffle(deck)
        if deck[-1].complaint == self.last_complaint:
            if len(deck) == 1:
                # A category switch can put the remaining template on both decks.
                refill = [s for s in SCENARIOS if s != deck[-1]
                          and (category is None or s.category == category)]
                self.rng.shuffle(refill)
                deck[:0] = refill
            deck[-1], deck[-2] = deck[-2], deck[-1]
        scenario = deck.pop()
        while True:
            systolic = self.rng.randint(*scenario.systolic)
            diastolic = self.rng.randint(scenario.diastolic[0], min(scenario.diastolic[1], systolic - 1))
            vitals = (
                systolic, diastolic, self.rng.randint(*scenario.pulse),
                self.rng.randint(*scenario.breathing),
                self.rng.randint(*scenario.temperature_tenths) / 10,
                self.rng.randint(*scenario.oxygen), self.rng.randint(*scenario.pain),
            )
            if vitals != self.last_vitals.get(scenario.complaint):
                break
        self.last_complaint = scenario.complaint
        self.last_vitals[scenario.complaint] = vitals
        return {
            "age": self.rng.randint(*scenario.age),
            "gender": self.rng.choice(("Male", "Female", "Not Provided")),
            "arrival_hour": self.rng.randrange(24),
            "chief_complaint": scenario.complaint,
            "encounter_id": f"DEMO-{uuid4().hex.upper()}",
            **dict(zip(VITAL_FIELDS, vitals)),
        }
