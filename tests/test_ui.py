"""Streamlit navigation and assessment smoke tests."""

import httpx
from streamlit.testing.v1 import AppTest


def test_assessment_and_session(monkeypatch):
    def backend(method, url, **kwargs):
        if url.endswith("/v1/status"):
            data = {"model_ready": True, "openai_configured": True, "acuity_classes": [1, 2, 3, 4]}
        elif url.endswith("/v1/performance"):
            data = {"leaderboard": [{"model": "XGBoost", "macro_f1": 0.75}]}
        else:
            assert kwargs["json"]["chief_complaint"]
            data = {
                "predicted_acuity": 3, "confidence": 0.8,
                "proba": {"1": 0.02, "2": 0.08, "3": 0.8, "4": 0.1},
                "shap_top": {"3": [{"feature": "age", "shap_value": 0.2}]},
                "llm_backend": "rule_based", "rationale": "Synthetic rationale.",
            }
        return httpx.Response(200, json=data, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", backend)
    app = AppTest.from_file("app.py").run()
    assert not app.exception
    assert app.sidebar.radio[0].options[1].endswith("Patient Triage")
    app.sidebar.radio[0].set_value("Patient Triage").run()
    next(button for button in app.button if button.label == "Run Triage Assessment").click().run()
    assert "chief complaint" in app.error[0].value
    next(button for button in app.button if button.label == "Generate Random Demo Patient").click().run()
    next(radio for radio in app.radio if radio.label == "Assessment mode").set_value("Detailed AI Triage")
    next(button for button in app.button if button.label == "Run Triage Assessment").click().run()
    assert not app.exception
    assert any("ESI 3" in heading.value for heading in app.subheader)
    app.sidebar.radio[0].set_value("Explainable AI").run()
    assert app.get("plotly_chart")
    app.sidebar.radio[0].set_value("Clinical Reasoning").run()
    assert any("rule-based" in message.value for message in app.info)
    app.sidebar.radio[0].set_value("Model Performance").run()
    assert app.get("plotly_chart")
    app.sidebar.button[0].click().run()
    next(button for button in app.button if button.label == "Confirm clear patient session").click().run()
    assert "result" not in app.session_state


def test_backend_offline(monkeypatch):
    def offline(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "request", offline)
    app = AppTest.from_file("app.py").run()
    assert not app.exception
    assert app.metric[0].value == "Unavailable"



def test_demo_generation_is_explicit_and_profile_survives_navigation(monkeypatch):
    def backend(method, url, **kwargs):
        assert method == "GET", "Generating a profile must not submit an assessment"
        return httpx.Response(200, json={}, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", backend)
    app = AppTest.from_file("app.py").run()
    app.sidebar.radio[0].set_value("Patient Triage").run()
    button = lambda: next(b for b in app.button if b.label == "Generate Random Demo Patient")
    app.session_state["result"] = {"shap_top": {"1": []}, "rationale": "Previous"}
    button().click().run()
    assert not app.exception
    assert "result" not in app.session_state
    assert any(message.value == "Synthetic Demo Patient" for message in app.info)
    assert any(message.value == "New synthetic patient generated." for message in app.success)
    assert app.number_input(key="patient_age").value is not None
    first = app.text_area(key="patient_chief_complaint").value
    encounter = app.text_input(key="patient_encounter_id").value
    assert encounter.startswith("DEMO-")
    app.sidebar.radio[0].set_value("Home").run()
    app.sidebar.radio[0].set_value("Patient Triage").run()
    assert app.text_input(key="patient_encounter_id").value == encounter
    assert app.text_area(key="patient_chief_complaint").value == first
    button().click().run()
    assert not app.exception
    assert app.text_area(key="patient_chief_complaint").value != first
    assert app.text_input(key="patient_encounter_id").value != encounter


def test_fast_prediction_can_add_reasoning_without_replacing_prediction(monkeypatch):
    calls = []
    probabilities = {"1": .05, "2": .25, "3": .60, "4": .10}

    def backend(method, url, **kwargs):
        route = httpx.URL(url).path
        if route == "/v1/status":
            body = {"model_ready": True, "openai_configured": True, "shap_installed": True}
        else:
            calls.append((route, kwargs["json"]))
            common = {"model_version": "test-version", "confidence": .60, "proba": probabilities}
            if route == "/v1/predict":
                body = {**common, "predicted": 3, "shap_output": "Uncalibrated raw model score",
                        "shap_top": {"3": [{"feature": "heart_rate", "shap_value": .3}],
                                     "4": [{"feature": "temperature", "shap_value": -.25}]}}
            else:
                body = {**common, "predicted_acuity": 3, "llm_backend": "rule_based",
                        "rationale": "Synthetic test rationale."}
        return httpx.Response(200, json=body, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", backend)
    app = AppTest.from_file("app.py").run()
    app.sidebar.radio[0].set_value("Patient Triage").run()
    next(b for b in app.button if b.label == "Generate Random Demo Patient").click().run()
    mode = next(r for r in app.radio if r.label == "Assessment mode")
    assert mode.value == "Detailed AI Triage"
    mode.set_value("Fast Prediction")
    next(b for b in app.button if b.label == "Run Triage Assessment").click().run()
    original = app.session_state["result"]
    app.sidebar.radio[0].set_value("Explainable AI").run()
    assert [t.label for t in app.tabs] == ["SHAP Chart", "Feature Table", "Explanation"]
    assert app.selectbox[0].value == "3"
    app.selectbox[0].set_value("4").run()
    assert any(metric.label == "Temperature" for metric in app.metric)
    app.sidebar.radio[0].set_value("Clinical Reasoning").run()
    assert len(calls) == 1
    next(b for b in app.button if b.label == "Generate Clinical Reasoning").click().run()
    assert calls[0] == ("/v1/predict", calls[1][1])
    assert calls[1][0] == "/v1/triage"
    assert app.session_state["result"] == original
    assert app.session_state["patient_reasoning"]["llm_backend"] == "rule_based"
    app.sidebar.radio[0].set_value("System Status").run()
    assert not app.exception and len(calls) == 2
    assert any(m.label == "Current reasoning source" and m.value == "Local fallback" for m in app.metric)
    app.sidebar.radio[0].set_value("About Project").run()
    assert not app.exception
    assert all("Patient Records" not in option for option in app.sidebar.radio[0].options)
