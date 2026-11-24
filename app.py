"""Streamlit dashboard for the emergency triage prototype."""

import os
from uuid import uuid4

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from synthetic_patients import CATEGORIES, DemoPatientGenerator

load_dotenv()
st.set_page_config(page_title="Emergency Healthcare AI", layout="wide")

DISCLAIMER = "Decision-support only. Not a diagnosis. Confirm with a licensed clinician."
LABELS = {1: "Critical", 2: "Emergent", 3: "Urgent", 4: "Less Urgent"}
COLORS = ["#b42318", "#b54708", "#936300", "#027a48"]
PAGES = (
    "Home", "Patient Triage", "Explainable AI", "Clinical Reasoning",
    "Model Performance", "System Status", "About Project",
)
VITALS = {
    "systolic_bp": ("Systolic blood pressure (mmHg)", 40, 300),
    "diastolic_bp": ("Diastolic blood pressure (mmHg)", 20, 200),
    "heart_rate": ("Heart rate (bpm)", 20, 300),
    "respiratory_rate": ("Respiratory rate (breaths/min)", 4, 60),
    "temperature": ("Temperature (°C)", 30.0, 43.0),
    "spo2": ("Oxygen saturation (%)", 50, 100),
    "pain_score": ("Pain score", 0, 10),
}
PATIENT_WIDGETS = (
    "age", "gender", "arrival_hour", "chief_complaint", "encounter_id",
    "record_type", *VITALS,
)


def request(path, payload=None, *, assessment_headers=None):
    url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/") + path
    token = os.getenv("FRONTEND_API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    headers.update(assessment_headers or {})
    response = httpx.request(
        "POST" if payload is not None else "GET", url, json=payload,
        headers=headers, timeout=30 if payload is not None else 3,
    )
    response.raise_for_status()
    return response.json()


def system_status():
    try:
        return request("/v1/status")
    except (httpx.HTTPError, ValueError):
        return None


def reset_patient():
    for key in list(st.session_state):
        if key.startswith("patient_") or key in ("result", "submitted_vitals"):
            del st.session_state[key]



def generate_demo_patient():
    generator = st.session_state.get("demo_patient_generator")
    if generator is None:
        generator = DemoPatientGenerator()
        st.session_state.demo_patient_generator = generator
    profile = generator.generate(st.session_state.demo_scenario)
    reset_patient()
    st.session_state.update({f"patient_{field}": value for field, value in profile.items()})
    st.session_state.patient_record_type = "Synthetic Demo"
    st.session_state.patient_is_demo = True
    st.session_state.patient_demo_notice = True


def patient_form():
    controls = st.columns(2)
    controls[0].selectbox(
        "Demo scenario", [None, *CATEGORIES], key="demo_scenario",
        format_func=lambda category: f"{category} Scenario" if category else "Random Scenario",
        help="Scenarios set the synthetic inputs. The model predicts acuity from the submitted values.",
    )
    controls[1].button("Generate Random Demo Patient", on_click=generate_demo_patient)
    controls[1].button("Clear patient data and result", on_click=reset_patient)
    is_demo = st.session_state.get("patient_is_demo", False)
    if st.session_state.pop("patient_demo_notice", False):
        st.success("New synthetic patient generated.")
    if is_demo:
        st.info("Synthetic Demo Patient")
    with st.form("triage"):
        st.subheader("Patient and arrival")
        cols = st.columns(3)
        age = cols[0].number_input("Age (years)", 0, 120, value=None, key="patient_age")
        gender = cols[1].selectbox(
            "Gender", ["Not Provided", "Male", "Female"], key="patient_gender"
        )
        hour = cols[2].number_input("Arrival hour", 0, 23, value=None, key="patient_arrival_hour")
        complaint = st.text_area("Chief complaint", max_chars=500, key="patient_chief_complaint")
        encounter = st.text_input("Encounter ID (optional)", max_chars=128, key="patient_encounter_id")
        st.subheader("Vital signs")
        cols = st.columns(2)
        vitals = {}
        for index, (field, (label, low, high)) in enumerate(VITALS.items()):
            vitals[field] = cols[index % 2].number_input(
                label, low, high, value=None, key=f"patient_{field}"
            )
        record_type = st.selectbox(
            "Record type", ["Synthetic Demo", "Authorised patient"],
            index=None, key="patient_record_type", disabled=is_demo,
        )
        writer_token = st.text_input(
            "Individual writer token (authorised patient records only)",
            type="password", key="patient_writer_token",
        )
        mode = st.radio("Assessment mode", ["Fast Prediction", "Detailed Triage"], horizontal=True)
        submitted = st.form_submit_button("Run assessment", type="primary")
    if submitted:
        st.session_state.pop("result", None)
        st.session_state.pop("submitted_vitals", None)
        if age is None or not complaint.strip():
            st.error("Enter an age and chief complaint.")
        elif record_type is None:
            st.error("Select the record type before saving.")
        else:
            payload = {
                "age": age, "gender": None if gender == "Not Provided" else gender,
                "arrival_hour": hour, "chief_complaint": complaint.strip(),
                "encounter_id": encounter.strip() or None, **vitals,
            }
            endpoint = "/v1/triage" if mode == "Detailed Triage" else "/v1/predict"
            signature = (endpoint, record_type, payload)
            previous = st.session_state.get("patient_submission")
            key = previous[1] if previous and previous[0] == signature else str(uuid4())
            st.session_state.patient_submission = (signature, key)
            headers = {"Idempotency-Key": key, "X-Record-Type": record_type}
            if record_type == "Authorised patient" and writer_token:
                headers["X-Patient-Writer-Token"] = writer_token
            try:
                with st.spinner("Running assessment…"):
                    st.session_state.result = request(endpoint, payload, assessment_headers=headers)
                    st.session_state.submitted_vitals = vitals
            except httpx.HTTPStatusError as exc:
                messages = {
                    401: "The dashboard could not authenticate with the backend.",
                    403: "Check your individual patient record writer credential.",
                    409: "This submission is already pending or failed. Clear the form to start a new assessment.",
                    422: "Review the assessment fields and submit again.",
                    503: "The model or Excel storage is unavailable. The assessment was not confirmed saved.",
                }
                st.error(messages.get(exc.response.status_code, "The assessment failed."))
            except (httpx.HTTPError, ValueError):
                st.error("The backend did not return an assessment. Check System Status.")
    if result := st.session_state.get("result"):
        if assessment_id := result.get("assessment_id"):
            st.success(f"Saved to Excel ? {assessment_id}")
        if result.get("assessment_replayed"):
            st.info("Showing the saved assessment. SHAP and narrative are available on the original result.")
        prediction(result)


def prediction(result):
    cls = result.get("predicted", result.get("predicted_acuity"))
    st.subheader(f"ESI {cls} · {LABELS.get(cls, 'Unknown')}")
    cards = st.columns(3)
    cards[0].metric("Model confidence", f"{result['confidence']:.1%}")
    cards[1].metric("Model version", result.get("model_version") or "Not available")
    latency = result.get("latency_ms")
    cards[2].metric("Latency", f"{latency} ms" if latency is not None else "Not available")
    frame = pd.DataFrame({"Class": [f"ESI {n}" for n in LABELS],
                          "Probability": [result["proba"].get(str(n)) for n in LABELS]})
    chart = px.bar(frame, x="Class", y="Probability", color="Class", color_discrete_sequence=COLORS)
    chart.update_yaxes(tickformat=".0%", range=[0, 1])
    chart.update_layout(showlegend=False)
    st.plotly_chart(chart, use_container_width=True)
    st.caption("Model confidence is the estimated probability of the selected class.")
    if result.get("safety_flags"):
        st.warning("Safety flags\n\n" + "\n".join(
            f"- {flag.replace('_', ' ')}" for flag in result["safety_flags"]
        ))
    if result.get("imputed_fields"):
        st.info("Fields imputed by the model: " + ", ".join(result["imputed_fields"]))
    observed = st.session_state.get("submitted_vitals", {})
    st.dataframe(
        pd.DataFrame([
            {"Vital": VITALS[field][0], "Observed": str(value) if value is not None else "Missing"}
            for field, value in observed.items()
        ]), hide_index=True, use_container_width=True,
    )
    with st.expander("Prediction JSON"):
        fields = ("predicted", "predicted_acuity", "confidence", "proba", "safety_flags",
                  "imputed_fields", "latency_ms", "model_version", "llm_backend")
        st.json({key: result[key] for key in fields if key in result})


def explain(result):
    available = {key: value for key, value in result.get("shap_top", {}).items() if value}
    if not available:
        st.info("Run a patient assessment with a model that provides SHAP explanations.")
        return
    predicted = str(result.get("predicted", result.get("predicted_acuity")))
    classes = sorted(available)
    current = st.selectbox("Acuity class", classes, index=classes.index(predicted)
                           if predicted in available else 0, format_func=lambda key: f"ESI {key}")
    frame = pd.DataFrame(available[current])
    frame = frame.iloc[frame["shap_value"].abs().argsort()[::-1]].reset_index(drop=True)
    frame["Direction"] = frame["shap_value"].map(lambda value: "Positive" if value >= 0 else "Negative")
    fig = px.bar(frame.iloc[::-1], x="shap_value", y="feature", color="Direction", orientation="h",
                 color_discrete_map={"Positive": "#087f8c", "Negative": "#b54708"},
                 labels={"shap_value": "SHAP contribution", "feature": "Feature"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.caption(
        "SHAP values show how features move the underlying model score. "
        "Direction indicates a higher or lower score for this class. Attribution alone cannot establish medical causation."
    )


def clinical_reasoning(result):
    source = result.get("llm_backend")
    if source is None:
        st.info("Run Detailed Triage to generate a rationale.")
        return
    if source == "openai":
        st.success(f"OpenAI reasoning used · {result.get('reasoning_model') or 'model name unavailable'}")
    else:
        st.info("Local rule-based reasoning used.")
        status = system_status() or {}
        if status.get("openai_configured"):
            st.caption("OpenAI did not provide a validated response for this assessment.")
    if result.get("rationale"):
        st.text(result["rationale"])
    for field, label in (
        ("abnormal_vitals", "Abnormal vital signs"), ("differential_risks", "Differential risks"),
        ("immediate_actions", "Immediate action considerations"), ("red_flags", "Red flags"),
    ):
        st.subheader(label)
        for item in result.get(field, []):
            st.text(f"• {item}")
        if not result.get(field):
            st.caption("No entries available.")
    if result.get("safety_note"):
        st.text(result["safety_note"])


def performance():
    try:
        report = request("/v1/performance")
    except (httpx.HTTPError, ValueError):
        st.info("Evaluation metrics are unavailable. Check backend status and authentication.")
        return
    if not report.get("leaderboard"):
        st.info("No evaluation metrics are attached to the loaded model.")
        return
    st.caption(report.get("dataset_source", "Dataset source unavailable for this artifact"))
    st.caption(report.get("protocol", "Evaluation protocol unavailable for this artifact"))
    cards = st.columns(3)
    cards[0].metric("Selected model", report.get("selected_model", "Unavailable"))
    cards[1].metric("Held-out cases", report.get("evaluated_rows", "Unavailable"))
    held_out = report.get("held_out", {})
    cards[2].metric("Held-out macro F1", f"{held_out['macro_f1']:.4f}"
                    if "macro_f1" in held_out else "Unavailable")
    frame = pd.DataFrame(report["leaderboard"])
    if frame["macro_f1"].max() >= .99:
        st.warning("Near-perfect results warrant a review of label shortcuts, duplicates, and dataset provenance.")
    st.subheader("Development cross-validation")
    frame.insert(0, "rank", range(1, len(frame) + 1))
    chart = px.bar(frame, x="model", y="macro_f1",
                   error_y="macro_f1_std" if "macro_f1_std" in frame else None,
                   color_discrete_sequence=["#087f8c"],
                   labels={"model": "Model", "macro_f1": "Mean validation macro F1"})
    st.plotly_chart(chart, use_container_width=True)
    number_columns = {
        column: st.column_config.NumberColumn(format="%.4f")
        for column in frame.select_dtypes(include="float")
    }
    st.dataframe(frame, hide_index=True, use_container_width=True, column_config=number_columns)
    if "baseline_macro_f1" in report:
        st.caption(f"Most-frequent baseline macro F1 ? {report['baseline_macro_f1']:.4f}")
    if "training_macro_f1" in frame:
        st.plotly_chart(px.bar(
            frame, x="model", y=["training_macro_f1", "macro_f1"], barmode="group",
            labels={"value": "Macro F1", "variable": "Partition"},
            title="Training versus validation",
        ), use_container_width=True)
        safety_columns = ["acuity1_recall", "critical_miss_rate", "under_triage_rate"]
        st.plotly_chart(px.bar(
            frame, x="model", y=safety_columns, barmode="group",
            labels={"value": "Rate", "variable": "Metric", "model": "Model"},
            title="Development safety metrics",
        ), use_container_width=True)
    if counts := report.get("dataset_audit", {}).get("class_counts"):
        st.plotly_chart(px.bar(
            x=[f"ESI {key}" for key in counts], y=list(counts.values()),
            labels={"x": "Acuity", "y": "Eligible cases"}, title="Dataset class distribution",
        ), use_container_width=True)
    st.subheader("Independent held-out evaluation")
    if matrix := report.get("confusion_matrix"):
        labels = [f"ESI {n}" for n in LABELS]
        st.plotly_chart(px.imshow(matrix, x=labels, y=labels, text_auto=True,
                                 labels={"x": "Predicted", "y": "Observed"},
                                 color_continuous_scale="Teal"), use_container_width=True)
    if per_class := report.get("per_class"):
        classes = pd.DataFrame(per_class)
        classes["class"] = classes["class"].map(lambda n: f"ESI {n}")
        st.plotly_chart(px.bar(
            classes, x="class", y=["precision", "recall", "f1-score"], barmode="group",
            hover_data=["support"], labels={"value": "Score", "variable": "Metric"},
            title="Class precision, recall and F1",
        ), use_container_width=True)
        st.dataframe(classes, hide_index=True, use_container_width=True)
    if learning := report.get("learning_curve"):
        st.plotly_chart(px.line(
            learning, x="training_rows", y=["training_macro_f1", "validation_macro_f1"],
            markers=True, labels={"value": "Macro F1", "variable": "Partition"},
            title="Learning curve on one fixed development fold",
        ), use_container_width=True)
    if calibration := report.get("calibration"):
        reliability = px.line(
            calibration["points"], x="probability", y="observed", color="class",
            line_dash="model", markers=True, title="Held-out probability calibration",
            labels={"probability": "Mean predicted probability", "observed": "Observed frequency"},
        )
        reliability.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line_dash="dot")
        st.plotly_chart(reliability, use_container_width=True)
        st.caption(calibration["method"])
        st.dataframe(pd.DataFrame({
            "Base": calibration["uncalibrated"], "Deployed": calibration["deployed"],
        }), use_container_width=True)
    if robustness := report.get("robustness"):
        stress = pd.DataFrame(robustness)
        st.plotly_chart(px.bar(
            stress, x="macro_f1", y="scenario", orientation="h", hover_data=["rows"],
            title="Supplementary stress tests",
        ), use_container_width=True)
        st.dataframe(stress, hide_index=True, use_container_width=True)
    if "protocol" not in report:
        st.info("Fold variability, learning curves, calibration curves and stress tests are unavailable for this artifact.")
    for field, label in (("held_out", "Held-out metrics"), ("safety", "Held-out safety metrics"),
                         ("fairness", "Demographic subgroup evaluation")):
        if report.get(field):
            with st.expander(label):
                data = report[field]
                if isinstance(data, list):
                    st.dataframe(data, hide_index=True, use_container_width=True)
                else:
                    st.json(data, expanded=False)


def status_cards():
    status = system_status()
    values = status or {}
    cards = st.columns(4)
    cards[0].metric("API", "Online" if status is not None else "Unavailable")
    cards[1].metric("Model", ("Ready" if values.get("model_ready") else "Unavailable"))
    cards[2].metric("Dataset rows", values.get("dataset_rows") or "Not available")
    cards[3].metric("Acuity classes", len(values["acuity_classes"])
                    if values.get("acuity_classes") else "Not available")
    st.caption(f"Model version · {values.get('model_version') or 'Not available'}")
    if values.get("algorithm"):
        st.caption(f"Active algorithm · {values['algorithm']}")
    if status is not None:
        st.caption("OpenAI configured" if values.get("openai_configured") else "Local reasoning configured")
    return values


def main():
    # Keep form values when Streamlit removes widgets on page navigation.
    for field in PATIENT_WIDGETS:
        key = f"patient_{field}"
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    if "demo_scenario" in st.session_state:
        st.session_state.demo_scenario = st.session_state.demo_scenario
    st.sidebar.title("Emergency Healthcare AI")
    page = st.sidebar.radio("Navigation", PAGES)
    st.sidebar.button("Delete current patient session", on_click=reset_patient)
    st.title("Emergency Healthcare AI" if page == "Home" else page)
    st.caption(DISCLAIMER)
    result = st.session_state.get("result", {})
    if page == "Home":
        st.write("Explore how emergency triage prioritises assessments by estimated acuity.")
        status_cards()
        st.subheader("Assessment workflow")
        st.write("Streamlit → FastAPI → tabular classifier → SHAP → OpenAI or local rationale")
        st.write("Training candidates include XGBoost, LightGBM, CatBoost, logistic regression and MLP.")
        st.info("Start in Patient Triage, then open Explainable AI or Clinical Reasoning to review the result.")
    elif page == "Patient Triage":
        patient_form()
    elif page == "Explainable AI":
        explain(result)
    elif page == "Clinical Reasoning":
        clinical_reasoning(result)
    elif page == "Model Performance":
        performance()
    elif page == "System Status":
        st.button("Refresh status")
        values = status_cards()
        if values:
            for path in ("/v1/health", "/v1/ready", "/v1/version"):
                try:
                    st.caption(f"{path} ? {request(path)}")
                except (httpx.HTTPError, ValueError):
                    st.caption(f"{path} ? Unavailable")
            if values.get("openai_configured"):
                st.caption(f"Configured OpenAI model ? {values.get('openai_model')}")
            st.caption("Local fallback available")
        else:
            st.code("uvicorn backend.main:app --port 8000")
        st.caption("OpenAI configuration is reported without making an external request.")
    else:
        st.write("An academic prototype for exploring tabular ML, SHAP, and structured clinical rationale.")
        st.write("This dataset and classifier use ESI levels 1–4. The standard ESI has five levels.")
        st.info("Use synthetic patients for demonstrations. Delete the current patient session when finished.")


if __name__ == "__main__":
    main()
