import requests
import streamlit as st

API_URL = "http://localhost:8000/classify"

EXAMPLES = {
    "Example 1 (regex)": "File data_1234.csv uploaded successfully by user User99.",
    "Example 2 (bert)": "Multiple login failures occurred on user 9052 account",
    "Example 3 (llm)": "Escalation rule execution failed for ticket ID 9807 - undefined escalation level.",
}

LABEL_COLORS = {
    "System Notification": "#28a745",
    "User Action": "#28a745",
    "Security Alert": "#dc3545",
    "Critical Error": "#dc3545",
    "HTTP Status": "#007bff",
    "Resource Usage": "#007bff",
    "Error": "#fd7e14",
    "Workflow Error": "#fd7e14",
    "Deprecation Warning": "#fd7e14",
}

LAYER_EXPLANATIONS = {
    "regex": "Matched a fixed pattern — no ML needed",
    "bert": "BERT embeddings + Logistic Regression",
    "llm": "Sent to Groq LLM (rare/complex log)",
}

st.set_page_config(page_title="Hybrid Log Classifier", layout="wide")

st.title("Hybrid Log Classifier")
st.caption("Classifies logs using Regex → BERT → LLM pipeline")

if "log_input" not in st.session_state:
    st.session_state.log_input = ""

left, right = st.columns([1, 1], gap="large")

with left:
    # Example buttons
    st.markdown("**Quick examples:**")
    ex_cols = st.columns(3)
    for i, (label, text) in enumerate(EXAMPLES.items()):
        if ex_cols[i].button(label, use_container_width=True):
            st.session_state.log_input = text

    log_message = st.text_area(
        "Enter a log message",
        value=st.session_state.log_input,
        height=160,
        key="log_input",
    )

    classify_clicked = st.button("Classify", type="primary", use_container_width=True)

with right:
    if classify_clicked:
        if not log_message.strip():
            st.warning("Please enter a log message.")
        else:
            with st.spinner("Classifying..."):
                try:
                    response = requests.post(
                        API_URL,
                        json={"log_message": log_message},
                        timeout=30,
                    )
                    response.raise_for_status()
                    data = response.json()

                    label = data["label"]
                    layer = data["layer"]
                    confidence = data["confidence"]
                    latency_ms = data["latency_ms"]

                    # Colored label badge
                    color = LABEL_COLORS.get(label, "#6c757d")
                    st.markdown(
                        f"""
                        <div style="
                            background-color: {color};
                            color: white;
                            padding: 16px 24px;
                            border-radius: 10px;
                            font-size: 1.5rem;
                            font-weight: bold;
                            text-align: center;
                            margin-bottom: 20px;
                        ">{label}</div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Metric cards
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Layer Used", layer.upper())
                    m2.metric("Confidence", f"{confidence:.2%}")
                    m3.metric("Latency (ms)", f"{latency_ms:.1f}")

                    # Layer explanation
                    explanation = LAYER_EXPLANATIONS.get(layer, "")
                    st.info(f"**{layer.upper()}:** {explanation}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the API. Make sure the backend is running on http://localhost:8000")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The API took too long to respond.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"API error: {e}")
    else:
        st.markdown(
            """
            <div style="
                color: #888;
                text-align: center;
                margin-top: 60px;
                font-size: 1.1rem;
            ">Results will appear here after classification.</div>
            """,
            unsafe_allow_html=True,
        )
