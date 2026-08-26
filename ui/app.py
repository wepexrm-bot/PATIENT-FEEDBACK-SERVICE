"""Streamlit UI for the patient feedback sentiment pipeline.

Run with:  streamlit run ui/app.py
Talks to the host-exposed APIs:
  - ingestion-api   http://localhost:8000
  - governance      http://localhost:8010
"""
import os

import requests
import streamlit as st

st.set_page_config(page_title="Patient Feedback Console", layout="wide")

INGESTION_URL = os.getenv("INGESTION_API_URL", "http://localhost:8000")
GOVERNANCE_URL = os.getenv("GOVERNANCE_API_URL", "http://localhost:8010")
REVIEW_TOKEN = os.getenv("REVIEW_TOKEN", "dev-reviewer-token")
LABELS = ["negative", "neutral", "positive"]


def api_health(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        return resp.ok
    except requests.RequestException:
        return False


def submit_feedback(base_url: str, patient_ref: str, source: str, text: str) -> dict:
    resp = requests.post(
        f"{base_url}/feedback",
        json={"patient_ref": patient_ref, "source": source, "text": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def list_review_queue(base_url: str) -> list[dict]:
    resp = requests.get(f"{base_url}/review-queue", timeout=10)
    resp.raise_for_status()
    return resp.json().get("items", [])


def submit_review(
    base_url: str, item_id: int, corrected_label: str, reviewer: str, token: str, role: str
) -> dict:
    resp = requests.post(
        f"{base_url}/review-queue/{item_id}",
        json={"corrected_label": corrected_label, "reviewer": reviewer},
        headers={"Authorization": f"Bearer {token}", "X-Reviewer-Role": role},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def compute_drift(base_url: str, model_version: str, token: str, role: str) -> dict:
    resp = requests.post(
        f"{base_url}/drift/{model_version}",
        headers={"Authorization": f"Bearer {token}", "X-Reviewer-Role": role},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def get_drift(base_url: str, model_version: str) -> dict:
    resp = requests.get(f"{base_url}/drift/{model_version}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def label_badge(label: str) -> None:
    color = {"positive": "green", "negative": "red", "neutral": "yellow"}.get(label, "blue")
    st.markdown(f"# :{color}[{label.capitalize()}]")


with st.sidebar:
    st.header("Connection")
    ingestion_url = st.text_input("Ingestion API", value=INGESTION_URL)
    governance_url = st.text_input("Governance API", value=GOVERNANCE_URL)
    token = st.text_input("Review token", value=REVIEW_TOKEN, type="password")
    role = st.selectbox("Role", ["reviewer", "admin"])
    st.divider()
    st.markdown("### Health")
    ing_ok = api_health(ingestion_url)
    gov_ok = api_health(governance_url)
    if ing_ok:
        st.success("ingestion-api: ok")
    else:
        st.error("ingestion-api: unreachable")
    if gov_ok:
        st.success("governance-service: ok")
    else:
        st.error("governance-service: unreachable")

analyze_tab, queue_tab, drift_tab = st.tabs(
    ["Analyze Feedback", "Review Queue", "Drift"]
)

with analyze_tab:
    st.header("Analyze Feedback")
    col_in, col_out = st.columns([1, 2])

    with col_in:
        patient_ref = st.text_input("Patient reference", value="pt-0001")
        source = st.selectbox("Source", ["portal", "sms", "kiosk"])
        text = st.text_area(
            "Comment",
            height=220,
            placeholder=(
                "The nurse Dr. Smith was kind, but I waited 3 hours. Call me at "
                "555-123-4567, SSN 123-45-6789. DOB 04/12/1980."
            ),
        )
        submitted = st.button("Analyze", type="primary", disabled=not text.strip())

    if submitted:
        with col_out:
            try:
                result = submit_feedback(ingestion_url, patient_ref, source, text)
            except requests.RequestException as exc:
                detail = "upstream error"
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    detail = exc.response.text
                st.error(f"Request failed: {detail}")
            else:
                st.markdown("### Prediction")
                label_badge(result.get("label", "unknown"))
                conf = float(result.get("confidence", 0.0))
                st.progress(int(conf * 100))
                st.caption(f"confidence: {conf:.3f}")
                meta = []
                if result.get("model_version"):
                    meta.append(f"model {result['model_version']}")
                if result.get("latency_ms") is not None:
                    meta.append(f"latency {result['latency_ms']:.1f} ms")
                if result.get("oov_score") is not None:
                    meta.append(f"out-of-vocab {result['oov_score']:.2f}")
                st.caption(" | ".join(meta))

                if result.get("flagged_for_review"):
                    reason = result.get("review_reason") or "unknown"
                    st.warning(f"Flagged for review — reason: {reason}")
                    if "domain-shift" in reason:
                        st.caption(
                            "This comment uses words outside the model's training "
                            "vocabulary, so its confidence isn't trustworthy — a human "
                            "should verify the sentiment."
                        )
                else:
                    st.success("Not flagged for review")

                st.markdown("#### Redaction")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Original**")
                    st.text(text)
                with col_b:
                    st.markdown("**Redacted**")
                    st.code(result.get("redacted_text", ""), language=None)

with queue_tab:
    st.header("Review Queue")
    if not token:
        st.warning("Enter a review token in the sidebar to review items.")
    refresh = st.button("Refresh queue")
    items = list_review_queue(governance_url) if (refresh or "queue_items" not in st.session_state) else st.session_state.queue_items
    st.session_state.queue_items = items

    if not items:
        st.info("No pending reviews.")
    else:
        st.dataframe(
            [
                {
                    "id": i["id"],
                    "prediction": i["prediction_id"],
                    "reason": i.get("reason"),
                    "label": i["label"],
                    "confidence": round(i["confidence"], 3),
                    "model_version": i["model_version"],
                    "oov_score": round(i["oov_score"], 3) if i.get("oov_score") is not None else "-",
                    "redacted_text": i["redacted_text"],
                }
                for i in items
            ],
            use_container_width=True,
        )

        ids = [i["id"] for i in items]
        selected_id = st.selectbox("Select item", ids, format_func=lambda x: f"#{x}")
        item = next(i for i in items if i["id"] == selected_id)

        st.markdown("**Redacted text**")
        st.code(item["redacted_text"], language=None)
        st.markdown(
            f"Model label: **{item['label']}** · confidence **{item['confidence']:.3f}** "
            f"· version **{item['model_version']}** · reason **{item.get('reason')}**"
        )
        if item.get("oov_score") is not None:
            st.caption(f"out-of-vocabulary ratio: {item['oov_score']:.3f}")

        reviewer = st.text_input("Reviewer name", value="console-reviewer")
        corrected = st.selectbox(
            "Corrected label", LABELS, index=LABELS.index(item["label"]) if item["label"] in LABELS else 0
        )
        col_ok, col_fix = st.columns(2)
        with col_ok:
            accept = st.button("Accept as-is", type="primary")
        with col_fix:
            fix = st.button("Correct")

        if accept or fix:
            chosen = item["label"] if accept else corrected
            try:
                submit_review(governance_url, selected_id, chosen, reviewer, token, role)
            except requests.RequestException as exc:
                st.error(f"Review failed: {exc}")
            else:
                st.success(f"Item #{selected_id} marked reviewed as '{chosen}'.")
                st.session_state.queue_items = list_review_queue(governance_url)
                st.rerun()

with drift_tab:
    st.header("Drift")
    model_version = st.text_input("Model version", value="v1.1.0")
    col_compute, col_view = st.columns(2)
    with col_compute:
        compute = st.button("Compute drift", type="primary")
    with col_view:
        view = st.button("View latest")

    metric = None
    if compute:
        if role != "admin":
            st.error("Computing drift requires the admin role.")
        else:
            try:
                metric = compute_drift(governance_url, model_version, token, role)
            except requests.RequestException as exc:
                st.error(f"Compute failed: {exc}")
    elif view:
        try:
            metric = get_drift(governance_url, model_version)
        except requests.RequestException as exc:
            st.error(f"No drift data for {model_version}: {exc}")

    if metric:
        st.markdown(f"### {metric['model_version']}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rolling accuracy", f"{metric['rolling_accuracy']:.3f}")
        col2.metric("Label PSI", f"{metric['label_psi']:.3f}")
        col3.metric("Degraded", metric["degraded"])
        col4.metric("Window size", metric.get("window_size", "-"))
        if metric.get("computed_at"):
            st.caption(f"computed at {metric['computed_at']}")