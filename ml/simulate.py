"""Reproducible synthetic experiment with overlapping simulated labels."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.preprocessing import INPUT_BOUNDS

SOURCE = "Synthetic experiment v1 (simulated labels)"
MEASUREMENTS = list(INPUT_BOUNDS)[2:]
# Scenario-specific mean shifts around a shared population baseline.
PRESENTATIONS = {
    "Breathing difficulty": (0, 0, 10, 6, 0, -5, -1),
    "Chest discomfort": (7, 4, 4, 2, 0, -1, 3),
    "Fever and fatigue": (-5, -3, 13, 3, 1.4, -2, 0),
    "Abdominal discomfort": (-3, -2, 3, 0, .2, 0, 2),
    "Injury after a fall": (1, 0, 4, 1, 0, 0, 2),
    "Dizziness": (-8, -4, -3, 0, 0, 0, -1),
    "Weakness and nausea": (-5, -3, 5, 1, .3, -1, 0),
    "Headache": (5, 3, 0, 0, .1, 0, 2),
}


def simulate(count=6000, seed=2026, shift=False):
    """Sample latent urgency first, then noisy observations of presentations."""
    rng = np.random.default_rng(seed)
    labels = rng.choice([1, 2, 3, 4], count, p=[.05, .18, .40, .37])
    names = list(PRESENTATIONS)
    profile_weights = np.array([
        [5, 4, 4, 2, 3, 2, 3, 2],
        [5, 4, 3, 3, 3, 2, 2, 2],
        [3, 3, 3, 4, 4, 3, 3, 3],
        [2, 2, 2, 4, 4, 3, 3, 4],
    ], dtype=float)
    profiles = [rng.choice(names, p=weights / weights.sum())
                for weights in profile_weights[labels - 1]]
    age = np.clip(rng.normal(47 + 3 * (3 - labels), 24, count), 1, 97).round()
    # Shared latent factors couple blood pressures, respiration, pulse, and oxygen.
    distress = rng.normal((3 - labels) * .55, .90 if not shift else 1.25)
    perfusion = rng.normal(0, 1, count)
    baseline = np.array([120, 73, 81, 17, 36.8, 97.2, 3.2])
    loads = np.array([-11, -6, 10, 3.2, .18, -2.0, 1.15])
    noise = rng.normal(size=(count, 7)) * [11, 7, 9, 2.3, .45, 1.4, 1.8]
    values = baseline + np.array([PRESENTATIONS[name] for name in profiles])
    values += distress[:, None] * loads + noise
    values[:, :2] += perfusion[:, None] * [10, 6] + (age[:, None] - 45) * [.13, .05]
    values[:, 1] = np.minimum(values[:, 1], values[:, 0] - 8)
    if shift:
        values[:, 5] -= 1.4
    for j, column in enumerate(MEASUREMENTS):
        values[:, j] = values[:, j].clip(*INPUT_BOUNDS[column])
    values[:, 1] = np.minimum(values[:, 1], values[:, 0] - 1)
    values = values.round(1)
    values[:, -1] = values[:, -1].round()
    missing_rate = .03 + .03 * (labels == 4) + (.10 if shift else 0)
    values[rng.random(values.shape) < missing_rate[:, None]] = np.nan
    frame = pd.DataFrame(values, columns=MEASUREMENTS)
    frame.insert(0, "arrival_hour", rng.integers(0, 24, count))
    frame.insert(0, "chief_complaint", profiles)
    frame.insert(0, "gender", rng.choice(["M", "F", None], count, p=[.49, .49, .02]))
    frame.insert(0, "age", age.astype(int))
    frame.insert(0, "encounter_id", [f"SIMV1-{seed}-{n:06d}" for n in range(count)])
    frame["acuity_target"] = labels
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_triage_v1.csv"))
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as output:
        simulate(args.count, args.seed).to_csv(output, index=False)


if __name__ == "__main__":
    main()
