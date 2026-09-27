"""Streamlit front end for the churn prediction API.

    streamlit run app/streamlit_app.py

Scoring happens only in the FastAPI service (POST /predict). Set the API
address in the sidebar or with the CHURN_API_URL environment variable.
"""

import streamlit as st

from api_client import (
    DEFAULT_API_URL,
    ApiError,
    ApiUnavailableError,
    ApiValidationError,
    ChurnApiClient,
)
from inputs import (
    CONTRACT_OPTIONS,
    DEFAULTS,
    GENDER_OPTIONS,
    INTERNET_ADDON_LABELS,
    INTERNET_OPTIONS,
    PAYMENT_OPTIONS,
    YES_NO,
    build_customer_payload,
    estimated_total_charges,
)

RISK_DISPLAY = {  # label + icon so risk is never conveyed by colour alone
    "LOW": (st.success, "✅", "Low risk"),
    "MEDIUM": (st.warning, "⚠️", "Medium risk"),
    "HIGH": (st.error, "🚨", "High risk"),
}

st.set_page_config(page_title="Churn Risk Scoring", page_icon="📉", layout="wide")

# --- Sidebar: API connection -----------------------------------------------------
with st.sidebar:
    st.header("API connection")
    api_url = st.text_input("API base URL", value=DEFAULT_API_URL)
    client = ChurnApiClient(api_url)
    try:
        health = client.health()
        st.success(f"Connected · model v{health['model_version']}")
    except ApiError as exc:
        st.warning(f"API not available: {exc}")
        st.caption("Start it with `uvicorn api.main:app` from the project root.")

st.title("Customer churn risk")
st.write(
    "Estimate how likely a customer is to cancel their subscription, so retention teams can "
    "prioritise outreach. Enter the customer's details and select **Predict**."
)


def yes_no(label: str, default: str, key: str) -> str:
    return st.radio(label, YES_NO, index=YES_NO.index(default), horizontal=True, key=key)


# --- Inputs ------------------------------------------------------------------------
col_profile, col_services, col_account = st.columns(3, gap="large")

with col_profile:
    st.subheader("Customer")
    gender = st.selectbox("Gender", GENDER_OPTIONS, index=GENDER_OPTIONS.index(DEFAULTS["gender"]))
    senior_citizen = yes_no("Senior citizen", DEFAULTS["senior_citizen"], "senior")
    partner = yes_no("Has partner", DEFAULTS["partner"], "partner")
    dependents = yes_no("Has dependents", DEFAULTS["dependents"], "dependents")

with col_services:
    st.subheader("Services")
    phone_service = yes_no("Phone service", DEFAULTS["phone_service"], "phone")
    multiple_lines = yes_no(
        "Multiple lines", DEFAULTS["multiple_lines"], "lines"
    ) if phone_service == "Yes" else "No"
    internet_service = st.selectbox(
        "Internet service", INTERNET_OPTIONS, index=INTERNET_OPTIONS.index(DEFAULTS["internet_service"])
    )
    addons = {}
    if internet_service != "No":
        for column, label in INTERNET_ADDON_LABELS.items():
            addons[column] = yes_no(label, DEFAULTS["addons"][column], column)
    else:
        st.caption("Internet add-ons are not applicable without internet service.")

with col_account:
    st.subheader("Account")
    contract = st.selectbox("Contract", CONTRACT_OPTIONS, index=CONTRACT_OPTIONS.index(DEFAULTS["contract"]))
    payment_method = st.selectbox(
        "Payment method", PAYMENT_OPTIONS, index=PAYMENT_OPTIONS.index(DEFAULTS["payment_method"])
    )
    paperless_billing = yes_no("Paperless billing", DEFAULTS["paperless_billing"], "paperless")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=DEFAULTS["tenure"], step=1)
    monthly_charges = st.number_input(
        "Monthly charges ($)", min_value=0.01, value=DEFAULTS["monthly_charges"], step=1.0, format="%.2f"
    )
    auto_total = st.checkbox("Estimate total charges as tenure × monthly", value=DEFAULTS["auto_total_charges"])
    if auto_total:
        total_charges = estimated_total_charges(tenure, monthly_charges)
        st.number_input("Total charges ($)", value=total_charges, disabled=True, format="%.2f")
    else:
        total_charges = st.number_input(
            "Total charges ($)", min_value=0.0, value=DEFAULTS["total_charges"], step=10.0, format="%.2f"
        )

payload = build_customer_payload(
    gender=gender, senior_citizen=senior_citizen, partner=partner, dependents=dependents,
    tenure=tenure, phone_service=phone_service, multiple_lines=multiple_lines,
    internet_service=internet_service, addons=addons, contract=contract,
    paperless_billing=paperless_billing, payment_method=payment_method,
    monthly_charges=monthly_charges, total_charges=total_charges,
)

st.divider()

# --- Prediction ------------------------------------------------------------------
if st.button("Predict", type="primary"):
    try:
        with st.spinner("Scoring customer…"):
            result = client.predict(payload)
    except ApiValidationError as exc:
        st.error("The API rejected these inputs:")
        for message in exc.messages:
            st.markdown(f"- {message}")
    except ApiUnavailableError as exc:
        st.error(f"{exc}. Check that the API is running and the URL in the sidebar is correct.")
    except ApiError as exc:
        st.error(f"Unexpected API error: {exc}")
    else:
        show_banner, icon, label = RISK_DISPLAY[result.risk_level]
        decision = "Flag for retention" if result.prediction == 1 else "Not flagged"
        show_banner(f"{label} — {decision}", icon=icon)

        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{result.churn_probability:.1%}")
        m2.metric("Prediction", "Likely to churn" if result.prediction == 1 else "Likely to stay")
        m3.metric("Risk level", result.risk_level)
        st.progress(min(max(result.churn_probability, 0.0), 1.0))

        st.caption(
            f"Customers are flagged when the probability is at least {result.threshold:.2f}; "
            "LOW means below that threshold, MEDIUM and HIGH are flagged. "
            f"Model version {result.model_version}. The score is a statistical estimate based on "
            "patterns in historical customers; it does not identify the cause of churn."
        )

with st.expander("Request sent to the API"):
    st.json(payload)
