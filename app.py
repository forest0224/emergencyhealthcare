"""Streamlit dashboard for the emergency triage prototype."""

import os
from pathlib import Path
from math import isclose, isfinite

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
COLORS = ["#DC2626", "#EA580C", "#CA8A04", "#15803D"]
CHART_COLORS = ["#168B9C", "#3B82F6", "#8B5CF6", "#14B8A6", "#F59E0B", "#64748B"]
NAVIGATION = {
    "Home": "space_dashboard", "Patient Triage": "medical_services",
    "Explainable AI": "insights",
    "Clinical Reasoning": "clinical_notes", "Model Performance": "monitoring",
    "System Status": "monitor_heart", "About Project": "info",
}
PAGES = tuple(NAVIGATION)
px.defaults.color_discrete_sequence = CHART_COLORS
STYLES = Path(__file__).parent / ".streamlit" / "soft-ui.css"
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
    *VITALS, "assessment_mode",
)


def request(path, payload=None):
    url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/") + path
    token = os.getenv("FRONTEND_API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = httpx.request(
        "POST" if payload is not None else "GET", url, json=payload,
        headers=headers, timeout=30 if payload is not None else 3,
    )
    response.raise_for_status()
    return response.json()



def plot(figure, *, height=350, margin=None):
    figure.update_layout(
        paper_bgcolor="#F4F7FB", plot_bgcolor="#F4F7FB", colorway=CHART_COLORS,
        font={"family": "Arial, sans-serif", "color": "#172B4D", "size": 13},
        height=height,
        margin=margin or {"l": 34, "r": 30, "t": 52 if figure.layout.title.text else 24, "b": 40},
        hoverlabel={"bgcolor": "#F0F4FA", "font_color": "#172B4D"},
        barcornerradius=4, legend={"bgcolor": "rgba(0,0,0,0)"},
    )
    figure.update_xaxes(gridcolor="#D4DEEA", zerolinecolor="#A7B7C9", automargin=True)
    figure.update_yaxes(gridcolor="#D4DEEA", zerolinecolor="#A7B7C9", automargin=True)
    st.plotly_chart(figure, use_container_width=True, theme=None,
                    config={"displaylogo": False, "responsive": True})

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
    st.session_state.patient_is_demo = True
    st.session_state.patient_demo_notice = True


def patient_form():
    is_demo = st.session_state.get("patient_is_demo", False)
    if st.session_state.pop("patient_demo_notice", False):
        st.success("New synthetic patient generated.")
    if is_demo:
        st.info("Synthetic Demo Patient", icon=":material/science:")
    with st.form("triage"):
        with st.container(key="card_patient_info"):
            st.subheader("Patient information")
            age_col, gender_col, hour_col = st.columns(3)
            age = age_col.number_input("Age (years)", 0, 120, value=None, key="patient_age")
            gender = gender_col.selectbox(
                "Gender", ["Not Provided", "Male", "Female"], key="patient_gender"
            )
            hour = hour_col.number_input("Arrival hour", 0, 23, value=None, key="patient_arrival_hour")
            complaint = st.text_area("Chief complaint", max_chars=500, key="patient_chief_complaint")
            with st.expander("Advanced information"):
                encounter = st.text_input("Encounter ID (optional)", max_chars=128, key="patient_encounter_id")
        with st.container(key="card_vitals"):
            st.subheader("Vital signs")
            st.caption("Leave measurements blank if they were not taken.")
            cols = st.columns(3)
            vitals = {}
            for index, (field, (label, low, high)) in enumerate(VITALS.items()):
                vitals[field] = cols[index % 3].number_input(
                    label, low, high, value=None, key=f"patient_{field}"
                )
        with st.container(key="card_assessment_controls"):
            st.subheader("Assessment controls")
            mode = st.radio("Assessment mode", ["Detailed AI Triage", "Fast Prediction"],
                            horizontal=True, key="patient_assessment_mode")
            submitted = st.form_submit_button(
                "Run Triage Assessment", type="primary", icon=":material/play_arrow:"
            )
            scenario, generate, clear = st.columns([1.7, 1.4, 1], vertical_alignment="bottom")
            scenario.selectbox(
                "Demo scenario", [None, *CATEGORIES], key="demo_scenario",
                format_func=lambda category: f"{category} Scenario" if category else "Random Scenario",
                help="The selected scenario sets demo inputs. The model predicts acuity from the submitted values.",
            )
            generate.form_submit_button(
                "Generate Random Demo Patient", on_click=generate_demo_patient,
                icon=":material/science:", use_container_width=True,
            )
            clear.form_submit_button("Reset Form", on_click=reset_patient,
                                     icon=":material/restart_alt:", use_container_width=True)
    if submitted:
        for key in ("result", "submitted_vitals", "patient_submitted_payload", "patient_reasoning",
                    "patient_shap_class"):
            st.session_state.pop(key, None)
        if age is None or not complaint.strip():
            st.error("Enter an age and chief complaint.")
        else:
            payload = {
                "age": age, "gender": None if gender == "Not Provided" else gender,
                "arrival_hour": hour, "chief_complaint": complaint.strip(),
                "encounter_id": encounter.strip() or None, **vitals,
            }
            endpoint = "/v1/triage" if mode == "Detailed AI Triage" else "/v1/predict"
            try:
                with st.spinner("Running assessment..."):
                    st.session_state.result = request(endpoint, payload)
                st.session_state.patient_submitted_payload = payload
                st.session_state.submitted_vitals = vitals
            except httpx.HTTPStatusError as exc:
                messages = {
                    401: "The dashboard could not authenticate with the backend.",
                    403: "The dashboard lacks inference permission.",
                    422: "Review the assessment fields and submit again.",
                    503: "The trained model is unavailable.",
                }
                st.error(messages.get(exc.response.status_code, "The assessment failed."))
            except (httpx.HTTPError, ValueError):
                st.error("The backend did not return an assessment. Check System Status.")
    if result := st.session_state.get("result"):
        prediction(result)


def prediction(result):
    cls = result.get("predicted", result.get("predicted_acuity"))
    severity = LABELS.get(cls, "Unknown")
    with st.container(key="card_result"):
        st.subheader(f"ESI {cls} - {severity}", anchor=False)
        if cls in LABELS:
            st.html(f'<span class="severity" style="--severity:{COLORS[cls - 1]}">'
                    f'{severity} acuity</span>')
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
    plot(chart)
    st.caption("Confidence estimates the selected class probability. Confirm the assessment with a licensed clinician.")
    if result.get("safety_flags"):
        st.error("Safety flags\n\n" + "\n".join(
            f"- {flag.replace('_', ' ')}" for flag in result["safety_flags"]
        ))
    if result.get("imputed_fields"):
        st.info("Fields imputed by the model: " + ", ".join(result["imputed_fields"]))
    with st.expander("Submitted vital signs"):
        observed = st.session_state.get("submitted_vitals", {})
        st.dataframe(pd.DataFrame([
            {"Vital": VITALS[field][0], "Observed": str(value) if value is not None else "Missing"}
            for field, value in observed.items()
        ]), hide_index=True, use_container_width=True)
    with st.expander("Prediction JSON"):
        fields = ("predicted", "predicted_acuity", "confidence", "proba", "safety_flags",
                  "imputed_fields", "latency_ms", "model_version", "llm_backend")
        st.json({key: result[key] for key in fields if key in result})


def explain(result):
    payload = result.get("shap_top")
    available = {}
    for key, values in (payload if isinstance(payload, dict) else {}).items():
        if key not in {"1", "2", "3", "4"} or not isinstance(values, list):
            continue
        entries = [entry for entry in values if isinstance(entry, dict)
                   and isinstance(entry.get("feature"), str)
                   and isinstance(entry.get("shap_value"), (int, float))
                   and isfinite(entry["shap_value"])]
        if entries:
            available[key] = entries
    if not available:
        st.info("No SHAP explanation is available. Run an assessment and check SHAP installation and model readiness in System Status.")
        return
    predicted = result.get("predicted", result.get("predicted_acuity"))
    classes = sorted(available)
    st.caption(f"Predicted acuity: ESI {predicted} - {LABELS.get(predicted, 'Unknown')}")
    current = st.selectbox("Acuity class", classes, index=classes.index(str(predicted))
                           if str(predicted) in available else 0, key="patient_shap_class",
                           format_func=lambda key: f"ESI {key} - {LABELS[int(key)]}")
    frame = pd.DataFrame(available[current]).sort_values(
        "shap_value", key=lambda column: column.abs(), ascending=False, kind="stable"
    )
    frame["Feature"] = frame["feature"].str.replace("_", " ").str.title()
    frame["Direction"] = frame["shap_value"].map(
        lambda value: "Positive" if value > 0 else "Negative" if value < 0 else "Zero"
    )
    for card, feature in zip(st.columns(3), frame.head(3).to_dict("records")):
        card.metric(feature["Feature"], f"{feature['shap_value']:+.3f}")
    chart, table, notes = st.tabs(["SHAP Chart", "Feature Table", "Explanation"])
    score = result.get("shap_output") or "Underlying model score (scale unavailable)"
    with chart:
        fig = px.bar(frame, x="shap_value", y="Feature", color="Direction", orientation="h",
                     color_discrete_map={"Positive": "#0F766E", "Negative": "#C2410C", "Zero": "#52647A"},
                     category_orders={"Feature": frame["Feature"].tolist()[::-1]},
                     labels={"shap_value": "Signed SHAP contribution"})
        fig.add_vline(x=0, line_width=1, line_color="#52647A")
        fig.update_traces(hovertemplate="%{y}: %{x:+.4f}<extra>%{fullData.name}</extra>")
        plot(fig, height=max(320, 150 + 36 * len(frame)), margin={"l": 48, "r": 34, "t": 34, "b": 48})
        st.caption(f"ESI {current} | {score}")
    with table:
        st.dataframe(frame[["Feature", "shap_value", "Direction"]], hide_index=True,
                     use_container_width=True, column_config={
                         "shap_value": st.column_config.NumberColumn("SHAP contribution", format="%+.4f")
                     })
    with notes:
        st.write(f"These attributions explain the {score.lower()}. Positive values increase the selected class score; negative values decrease it.")
        st.write("The displayed prediction may use calibrated probabilities. Feature contributions do not establish medical causation.")


def matches_assessment(original, detailed):
    original_proba = original.get("proba", {})
    detailed_proba = detailed.get("proba", {})
    return (
        original.get("model_version") == detailed.get("model_version")
        and original.get("predicted") == detailed.get("predicted_acuity")
        and original_proba.keys() == detailed_proba.keys()
        and all(isclose(value, detailed_proba[key], rel_tol=1e-7, abs_tol=1e-9)
                for key, value in original_proba.items())
    )


def clinical_reasoning(result, status):
    reasoning = st.session_state.get("patient_reasoning") or result
    source = reasoning.get("llm_backend")
    if source is None:
        payload = st.session_state.get("patient_submitted_payload")
        if not result or not payload:
            st.info("Complete an assessment in Patient Triage to review its clinical reasoning.")
            return
        st.info("This Fast Prediction has no clinical reasoning yet.")
        st.caption("This action reruns detailed triage on the submitted inputs. The original prediction stays in view.")
        if not st.button("Generate Clinical Reasoning", type="primary"):
            return
        try:
            with st.spinner("Generating clinical reasoning..."):
                reasoning = request("/v1/triage", payload)
        except (httpx.HTTPError, ValueError):
            st.error("Clinical reasoning is unavailable. Your original prediction is still available.")
            return
        if not matches_assessment(result, reasoning):
            st.warning("The model or prediction changed. Run a new Detailed AI Triage assessment to review a matching rationale.")
            return
        if not reasoning.get("llm_backend"):
            st.error("The backend did not return clinical reasoning.")
            return
        st.session_state.patient_reasoning = reasoning
        source = reasoning["llm_backend"]
    result = reasoning
    with st.container(key="card_reasoning_source"):
        st.subheader("Reasoning source")
        if source == "openai":
            st.success(f"OpenAI reasoning used: {result.get('reasoning_model') or 'model name unavailable'}")
        else:
            st.info("Local rule-based reasoning used.")
            if (status or {}).get("openai_configured"):
                st.caption("OpenAI did not provide a validated response for this assessment.")
    with st.container(key="card_reasoning_summary"):
        st.subheader("Clinical summary")
        st.text(result.get("rationale") or "Summary unavailable for this assessment.")
    for index, (field, label) in enumerate((
        ("abnormal_vitals", "Abnormal vital signs"), ("differential_risks", "Differential risks"),
        ("immediate_actions", "Immediate action considerations"), ("red_flags", "Red flags"),
    )):
        with st.container(key=f"card_reasoning_{index}"):
            st.subheader(label)
            for item in result.get(field, []):
                if field == "red_flags":
                    st.error(item)
                else:
                    st.text(f"- {item}")
            if not result.get(field):
                st.caption("No entries available.")
    with st.container(key="card_safety_notes"):
        st.subheader("Safety notes")
        st.text(result.get("safety_note") or "No additional safety notes recorded.")


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
    overview, compare, confusion, safety_tab, calibration_tab = st.tabs([
        "Overview", "Model Comparison", "Confusion Matrix", "Safety Metrics", "Calibration and Fairness"
    ])
    held = report.get("held_out") or {}
    safety = report.get("safety") or {}
    frame = pd.DataFrame(report["leaderboard"])
    cases = report.get("evaluated_rows", "Unavailable")
    with overview:
        metric_grid([
            ("Selected model", report.get("selected_model", "Unavailable")),
            ("Held-out Macro F1", metric_value(held.get("macro_f1"), ".4f")),
            ("Accuracy", metric_value(held.get("accuracy"))),
            ("Macro precision", metric_value(held.get("macro_precision"))),
            ("Macro recall", metric_value(held.get("macro_recall"))),
            ("Under-triage rate", metric_value(safety.get("under_triage_rate_overall"))),
        ])
        st.caption(f"Independent evaluation cases: {cases}")
        with st.expander("Additional evaluation metrics"):
            if held:
                st.json(held, expanded=False)
            else:
                st.info("Additional evaluation metrics are unavailable.")
            if counts := report.get("dataset_audit", {}).get("class_counts"):
                plot(px.bar(x=[f"ESI {key}" for key in counts], y=list(counts.values()),
                            labels={"x": "Acuity", "y": "Eligible cases"}, title="Dataset class distribution"),
                     height=320)
    with compare:
        st.subheader("Development cross-validation")
        if frame["macro_f1"].max() >= .99:
            st.warning("Near-perfect scores warrant a review of label shortcuts, duplicates and provenance.")
        chart = px.bar(
            frame, x="macro_f1", y="model", orientation="h",
            error_x="macro_f1_std" if "macro_f1_std" in frame else None,
            labels={"macro_f1": "Mean validation Macro F1", "model": "Model"},
            color_discrete_sequence=["#0F766E"],
        )
        chart.update_xaxes(range=[0, max(1, frame["macro_f1"].max()
                                        + frame.get("macro_f1_std", pd.Series([0])).fillna(0).max())])
        chart.update_yaxes(categoryorder="array", categoryarray=frame["model"].tolist()[::-1])
        plot(chart, height=max(310, 130 + 38 * len(frame)))
        columns = ["model", "macro_f1", "macro_f1_std", "training_macro_f1", "validation_accuracy"]
        shown = frame[[name for name in columns if name in frame]]
        st.dataframe(shown, hide_index=True, use_container_width=True, column_config={
            "macro_f1": st.column_config.NumberColumn("Mean Macro F1", format="%.4f"),
            "macro_f1_std": st.column_config.NumberColumn("CV standard deviation", format="%.4f"),
        })
        if "baseline_macro_f1" in report:
            st.caption(f"Most-frequent baseline Macro F1: {report['baseline_macro_f1']:.4f}")
        if learning := report.get("learning_curve"):
            with st.expander("Learning curve"):
                plot(px.line(learning, x="training_rows", y=["training_macro_f1", "validation_macro_f1"],
                             markers=True, title="One fixed development fold", labels={"value": "Macro F1"}))
    with confusion:
        st.subheader("Independent confusion matrix")
        if matrix := report.get("confusion_matrix"):
            labels = [f"ESI {n}" for n in LABELS]
            figure = px.imshow(matrix, x=labels, y=labels, text_auto=True, aspect="auto",
                               labels={"x": "Predicted", "y": "Observed"}, color_continuous_scale="Teal")
            plot(figure, height=340)
        else:
            st.info("The confusion matrix is unavailable.")
        if per_class := report.get("per_class"):
            table = pd.DataFrame(per_class)
            table["class"] = table["class"].map(lambda number: f"ESI {number}")
            st.dataframe(table, hide_index=True, use_container_width=True)
    with safety_tab:
        st.subheader("Independent safety metrics")
        metric_grid([
            ("Critical miss rate", metric_value(safety.get("critical_miss_rate"))),
            ("High-acuity under-triage", metric_value(safety.get("under_triage_rate_overall"))),
            ("Over-triage rate", metric_value(safety.get("over_triage_rate_overall"))),
            ("Critical patient recall", metric_value(held.get("acuity1_recall"))),
        ])
        st.caption(f"Independent evaluation cases: {cases}. See class support in Confusion Matrix.")
        st.write("Critical misses are ESI 1 cases predicted as ESI 3 or 4. Under-triage measures ESI 1 and 2 cases "
                 "assigned a less urgent category. Over-triage measures all cases assigned a more urgent category. "
                 "Critical recall is the proportion of ESI 1 cases predicted as ESI 1.")
        if robustness := report.get("robustness"):
            with st.expander("Supplementary stress tests"):
                st.dataframe(pd.DataFrame(robustness), hide_index=True, use_container_width=True)
    with calibration_tab:
        st.subheader("Held-out calibration and subgroup evaluation")
        if calibration := report.get("calibration"):
            if points := calibration.get("points"):
                reliability = px.line(
                    points, x="probability", y="observed", color="class",
                    line_dash="model", markers=True,
                    labels={"probability": "Mean predicted probability", "observed": "Observed frequency"},
                )
                reliability.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line_dash="dot")
                reliability.update_xaxes(range=[0, 1])
                reliability.update_yaxes(range=[0, 1])
                plot(reliability)
            st.caption(calibration.get("method", "Calibration method unavailable."))
            st.dataframe(pd.DataFrame({
                "Base": calibration.get("uncalibrated", {}),
                "Deployed": calibration.get("deployed", {}),
            }), use_container_width=True)
        else:
            st.info("Calibration metrics are unavailable.")
        if fairness := report.get("fairness"):
            st.subheader("Demographic fairness")
            st.dataframe(pd.DataFrame(fairness), hide_index=True, use_container_width=True)
        else:
            st.info("Demographic subgroup metrics are unavailable.")


def metric_value(value, format_spec=".2%"):
    return format(value, format_spec) if value is not None else "Unavailable"


def metric_grid(items):
    for index in range(0, len(items), 3):
        for card, (name, value) in zip(st.columns(3), items[index:index + 3]):
            card.metric(name, value)


def status_cards(status):
    values = status or {}
    cards = st.columns(4)
    cards[0].metric("API", "Online" if status is not None else "Unavailable")
    cards[1].metric("Model", "Ready" if values.get("model_ready") else "Unavailable")
    rows = values.get("dataset_rows")
    cards[2].metric("Dataset rows", f"{rows:,}" if rows is not None else "Unavailable")
    cards[3].metric("Acuity classes", len(values["acuity_classes"])
                    if values.get("acuity_classes") else "Unavailable")


def home(status):
    values = status or {}
    with st.container(key="hero"):
        introduction, actions = st.columns([2.2, 1])
        with introduction:
            st.title("Emergency Healthcare AI", anchor=False)
            st.write("Assess emergency department acuity, review the factors behind a prediction, "
                     "and keep an auditable record of the encounter.")
        with actions:
            with st.container(key="home_actions"):
                st.button("Start patient triage", type="primary", icon=":material/medical_services:",
                          on_click=set_page, args=("Patient Triage",), use_container_width=True)
                st.button("View model performance", icon=":material/monitoring:",
                          on_click=set_page, args=("Model Performance",), use_container_width=True)
    status_cards(status)
    with st.container(key="card_workflow"):
        st.subheader("Assessment workflow")
        for column, (title, description) in zip(st.columns(3), (
            ("Assess", "Streamlit sends the submitted demographics and vital signs to FastAPI."),
            ("Predict", "The trained classifier estimates acuity. SHAP shows contributing features."),
            ("Review", "Detailed triage provides OpenAI or local reasoning for the current session."),
        )):
            column.markdown(f"**{title}**")
            column.write(description)
        if values.get("models"):
            st.caption("Evaluated models: " + ", ".join(values["models"]))


def system_page(status):
    st.button("Refresh status", icon=":material/refresh:")
    status_cards(status)
    if status is None:
        st.error("Could not reach FastAPI.")
        st.code("uvicorn backend.main:app --port 8000")
        return
    current = st.session_state.get("result") or {}
    reasoning = st.session_state.get("patient_reasoning") or current
    source = reasoning.get("llm_backend")
    metric_grid([
        ("OpenAI configuration", "Configured" if status.get("openai_configured") else "Not configured"),
        ("Local fallback", "Available" if status.get("local_fallback_available") else "Unavailable"),
        ("SHAP package", "Installed" if status.get("shap_installed") else "Unavailable"),
        ("Current SHAP explanation", "Generated" if current.get("shap_top") else "Unavailable"),
        ("Current reasoning source", {"openai": "OpenAI", "rule_based": "Local fallback"}.get(source, "Unavailable")),
        ("API version", status.get("api_version") or "Unknown"),
        ("Model version", status.get("model_version") or "Unavailable"),
    ])
    st.caption("Status checks use local service information and current session results.")


def about():
    sections = (
        ("Purpose", "Explore how a decision-support prototype estimates acuity and explains its results."),
        ("Architecture", "Streamlit sends an assessment to FastAPI. The classifier and SHAP run locally. "
         "Detailed reasoning uses OpenAI when available, with local rules as fallback. Assessments remain in the current session."),
        ("Technology", "Streamlit, FastAPI, scikit-learn, gradient boosting, SHAP and Plotly."),
        ("Limitations", "The dataset uses ESI levels 1?4. The standard ESI has five levels. "
         "Project evaluation does not establish clinical reliability. Use synthetic patients for demonstrations."),
        ("Documentation", "See README.md for setup, evaluation, data provenance and session behavior. "
         "The FastAPI service exposes its OpenAPI documentation at /docs."),
    )
    for index, (heading, paragraph) in enumerate(sections):
        with st.container(key=f"card_about_{index}"):
            st.subheader(heading)
            st.write(paragraph)


def set_page(page):
    st.session_state.navigation = page


@st.dialog("Clear patient session")
def confirm_clear():
    st.write("Clear the current form, prediction, SHAP explanation and clinical reasoning.")
    if st.button("Confirm clear patient session", type="primary", on_click=reset_patient):
        st.rerun()
    if st.button("Cancel"):
        st.rerun()


def main():
    for field in PATIENT_WIDGETS:
        key = f"patient_{field}"
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    if "patient_shap_class" in st.session_state:
        st.session_state.patient_shap_class = st.session_state.patient_shap_class
    if "demo_scenario" in st.session_state:
        st.session_state.demo_scenario = st.session_state.demo_scenario
    st.html(str(STYLES))
    status = system_status()
    with st.sidebar:
        with st.container(key="brand"):
            st.html('''<div class="brand-lockup">
                <span class="brand-symbol" aria-hidden="true">+</span>
                <div><strong>Emergency Healthcare AI</strong>
                <small>Clinical Decision Support</small></div>
                </div>''')
        page = st.radio("Navigation", PAGES, key="navigation", label_visibility="collapsed",
                        format_func=lambda name: f":material/{NAVIGATION[name]}: {name}")
        values = status or {}
        st.caption(f"FastAPI: {'Online' if status is not None else 'Unavailable'}")
        st.caption(f"Model: {'Ready' if values.get('model_ready') else 'Unavailable'}")
        st.caption(f"API version: {values.get('api_version') or 'Unavailable'}")
        if st.button("Clear Patient Session", icon=":material/delete_sweep:", use_container_width=True):
            confirm_clear()
    if page != "Home":
        st.title(page, anchor=False)
    st.caption(DISCLAIMER)
    result = st.session_state.get("result", {})
    handlers = {
        "Home": lambda: home(status), "Patient Triage": patient_form,
        "Explainable AI": lambda: explain(result), "Clinical Reasoning": lambda: clinical_reasoning(result, status),
        "Model Performance": performance, "System Status": lambda: system_page(status), "About Project": about,
    }
    handlers[page]()


if __name__ == "__main__":
    main()
