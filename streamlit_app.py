import json

import requests
import streamlit as st


API_URL = "http://localhost:8000"


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Support Ticket Triage",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        h1 {
            font-size: 2rem !important;
            font-weight: 600 !important;
            margin-bottom: 2rem;
        }

        h2, h3 {
            font-weight: 600 !important;
        }

        .stTextInput input,
        .stTextArea textarea {
            border-radius: 6px;
        }

        .stButton button {
            width: 100%;
            border-radius: 6px;
            font-weight: 500;
        }

        .result-card {
            padding: 1rem;
            border: 1px solid #e5e5e5;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# API Functions
# --------------------------------------------------

def check_api_health() -> bool:
    """Check whether the FastAPI backend is available."""

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def submit_ticket(payload: dict) -> dict:
    """Send ticket to the FastAPI triage endpoint."""

    response = requests.post(
        f"{API_URL}/triage",
        json=payload,
        timeout=120,
    )

    if response.status_code != 200:

        try:
            error_detail = response.json().get(
                "detail",
                "Unknown API error",
            )
        except ValueError:
            error_detail = response.text

        raise RuntimeError(error_detail)

    return response.json()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("Support Ticket Triage")


# --------------------------------------------------
# Backend Status
# --------------------------------------------------

api_available = check_api_health()

if not api_available:
    st.error(
        "Unable to connect to the triage API. "
        "Make sure the FastAPI server is running on port 8000."
    )


# --------------------------------------------------
# Ticket Input
# --------------------------------------------------

with st.form("ticket_form"):

    col1, col2 = st.columns(2)

    with col1:

        ticket_id = st.text_input(
            "Ticket ID",
            placeholder="e.g. TKT-1001",
        )

        subject = st.text_input(
            "Subject",
            placeholder="Enter ticket subject",
        )

        company = st.text_input(
            "Company",
            placeholder="Enter company name",
        )

    with col2:

        plan_tier = st.text_input(
            "Plan Tier",
            placeholder="e.g. Enterprise",
        )

        product = st.text_input(
            "Product",
            placeholder="e.g. DataBridge Pro",
        )

    body = st.text_area(
        "Ticket Description",
        placeholder="Enter the customer's issue or request",
        height=200,
    )

    submitted = st.form_submit_button(
        "Analyze Ticket",
        type="primary",
    )


# --------------------------------------------------
# Submit Ticket
# --------------------------------------------------

if submitted:

    if not api_available:
        st.error("The API server is currently unavailable.")

    elif not body.strip():
        st.error("Ticket description is required.")

    else:

        payload = {
            "ticket_id": ticket_id.strip() or None,
            "subject": subject.strip(),
            "body": body.strip(),
            "company": company.strip() or None,
            "plan_tier": plan_tier.strip() or None,
            "product": product.strip() or None,
        }

        try:

            with st.spinner("Analyzing support ticket..."):

                result = submit_ticket(payload)

            st.session_state["triage_result"] = result
            st.session_state["ticket_payload"] = payload

        except requests.Timeout:

            st.error(
                "The request timed out. "
                "The triage pipeline may still be processing."
            )

        except requests.RequestException as exc:

            st.error(
                f"Unable to connect to the API: {exc}"
            )

        except RuntimeError as exc:

            st.error(
                f"Triage request failed: {exc}"
            )


# --------------------------------------------------
# Results
# --------------------------------------------------

if "triage_result" in st.session_state:

    result = st.session_state["triage_result"]

    st.divider()

    st.subheader("Triage Result")

    # --------------------------------------------------
    # Classification Summary
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Product Area",
            result.get(
                "product_area",
                "N/A",
            ),
        )

    with col2:
        st.metric(
            "Issue Category",
            result.get(
                "category",
                "N/A",
            ),
        )

    with col3:
        st.metric(
            "Urgency",
            result.get(
                "urgency",
                "N/A",
            ),
        )

    # --------------------------------------------------
    # Details
    # --------------------------------------------------

    tab1, tab2, tab3 = st.tabs(
        [
            "Analysis",
            "Recommended Response",
            "JSON",
        ]
    )

    # ---------------- Analysis ----------------

    with tab1:

        if result.get("reasoning"):

            st.markdown("### Classification Reasoning")

            st.write(
                result["reasoning"]
            )

        if result.get("routing"):

            st.markdown("### Routing")

            st.write(
                result["routing"]
            )

        if result.get("team"):

            st.markdown("### Assigned Team")

            st.write(
                result["team"]
            )

        if result.get("priority_reasoning"):

            st.markdown("### Priority Reasoning")

            st.write(
                result["priority_reasoning"]
            )

        if result.get("knowledge_base_matches"):

            st.markdown(
                "### Relevant Knowledge Base Information"
            )

            matches = result[
                "knowledge_base_matches"
            ]

            for item in matches:

                if isinstance(item, dict):

                    title = item.get(
                        "title",
                        "Knowledge Base Match",
                    )

                    st.markdown(
                        f"**{title}**"
                    )

                    if item.get("content"):
                        st.write(
                            item["content"]
                        )

                else:
                    st.write(item)

    # ---------------- Response ----------------

    with tab2:

        response_text = result.get(
            "draft_response",
            result.get(
                "response",
                "",
            ),
        )

        if response_text:

            st.text_area(
                "Draft Response",
                value=response_text,
                height=300,
            )

        else:

            st.info(
                "No draft response was returned."
            )

    # ---------------- JSON ----------------

    with tab3:

        st.json(result)

        json_output = json.dumps(
            result,
            indent=2,
        )

        st.download_button(
            label="Download Result JSON",
            data=json_output,
            file_name="triage_result.json",
            mime="application/json",
        )