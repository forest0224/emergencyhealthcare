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
    assert app.sidebar.radio[0].options[1] == "Patient Triage"
    app.sidebar.radio[0].set_value("Patient Triage").run()
    next(button for button in app.button if button.label == "Run assessment").click().run()
    assert "chief complaint" in app.error[0].value
    next(button for button in app.button if button.label == "Load synthetic demo patient").click().run()
    next(radio for radio in app.radio if radio.label == "Assessment mode").set_value("Detailed Triage")
    next(button for button in app.button if button.label == "Run assessment").click().run()
    assert not app.exception
    assert any("ESI 3" in heading.value for heading in app.subheader)
    app.sidebar.radio[0].set_value("Explainable AI").run()
    assert app.get("plotly_chart")
    app.sidebar.radio[0].set_value("Clinical Reasoning").run()
    assert any("rule-based" in message.value for message in app.info)
    app.sidebar.radio[0].set_value("Model Performance").run()
    assert app.get("plotly_chart")
    app.sidebar.button[0].click().run()
    assert "result" not in app.session_state


def test_backend_offline(monkeypatch):
    def offline(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "request", offline)
    app = AppTest.from_file("app.py").run()
    assert not app.exception
    assert app.metric[0].value == "Unavailable"
