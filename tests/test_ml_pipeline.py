"""
tests/test_ml_pipeline.py
==========================
ML pipeline tests on a synthetic 200-row dataset.
"""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def _synthetic_csv(tmp_path, n=200, seed=42):
    rng = np.random.default_rng(seed)
    complaints = ["chest pain", "shortness of breath", "fever", "trauma", "headache"]
    df = pd.DataFrame({
        "encounter_id":     [f"ENC{i:05d}" for i in range(n)],
        "age":              rng.integers(1, 90, n).astype(float),
        "gender":           rng.choice(["M", "F"], n),
        "chief_complaint":  rng.choice(complaints, n),
        "arrival_hour":     rng.integers(0, 24, n),
        "systolic_bp":      rng.integers(80, 200, n).astype(float),
        "diastolic_bp":     rng.integers(50, 110, n).astype(float),
        "heart_rate":       rng.integers(50, 140, n).astype(float),
        "respiratory_rate": rng.integers(10, 30, n).astype(float),
        "temperature":      rng.uniform(36.0, 39.5, n),
        "spo2":             rng.integers(88, 100, n).astype(float),
        "pain_score":       rng.integers(0, 10, n).astype(float),
        "acuity_target":    rng.choice([1, 2, 3, 4], n,
                                        p=[0.02, 0.08, 0.40, 0.50]),
    })
    path = str(tmp_path / "synth.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth_csv(tmp_path_factory):
    return _synthetic_csv(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="module")
def features(synth_csv):
    from ml.preprocessing import build_features
    return build_features(synth_csv)


# ---------------------------------------------------------------------------
# feature_engineering
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_shape(self, features):
        X, y, meta = features
        assert X.shape[1] > 20, f"Expected >20 features, got {X.shape[1]}"

    def test_target_values(self, features):
        _, y, _ = features
        assert set(y.unique()).issubset({1, 2, 3, 4})

    def test_no_nan(self, features):
        X, _, _ = features
        assert not X.isnull().any().any()

    def test_meta_keys(self, features):
        _, _, meta = features
        assert "feature_names" in meta
        assert len(meta["feature_names"]) > 0


# ---------------------------------------------------------------------------
# tabular_models
# ---------------------------------------------------------------------------

class TestTrainAndEvaluate:
    def test_returns_macro_f1_and_under_triage(self, features, tmp_path):
        """train_and_evaluate result dict contains required metric keys."""
        from ml import train as tm_mod
        from sklearn.linear_model import LogisticRegression

        X, y, _ = features
        orig = tm_mod._make_models
        tm_mod._make_models = lambda random_state=42: {
            "LR": LogisticRegression(
                max_iter=200, class_weight="balanced",
                multi_class="multinomial", solver="lbfgs", random_state=42,
            )
        }
        try:
            result = tm_mod.train_and_evaluate(
                X, y, n_splits=2, output_dir=str(tmp_path)
            )
        finally:
            tm_mod._make_models = orig

        lb = result["leaderboard"]
        assert "macro_f1" in lb.columns
        # under_triage_rate_overall lives in per-fold metrics
        fold_metrics = result["all_metrics"]["LR"]
        assert all("under_triage_rate" in fm for fm in fold_metrics)


# ---------------------------------------------------------------------------
# safety_metrics
# ---------------------------------------------------------------------------

class TestSafetyMetrics:
    def test_keys_present(self):
        from ml.safety_metrics import compute_safety_metrics
        y_true = np.array([1, 1, 2, 3, 3, 4])
        y_pred = np.array([1, 2, 2, 3, 4, 4])
        m = compute_safety_metrics(y_true, y_pred)
        for key in ("critical_miss_rate", "under_triage_rate_overall",
                    "macro_f1", "high_acuity_sensitivity"):
            assert key in m, f"Missing: {key}"


# ---------------------------------------------------------------------------
# PII scrubber (standalone — no ML needed)
# ---------------------------------------------------------------------------

class TestPIIScrubber:
    def test_strips_name_and_mrn(self):
        from backend.security import scrub_free_text
        result = scrub_free_text("Dr. Smith MRN 12345")
        assert "Smith" not in result
        assert "12345" not in result

    def test_preserves_clinical_text(self):
        from backend.security import scrub_free_text
        text = "chest pain and dyspnoea"
        assert scrub_free_text(text) == text
