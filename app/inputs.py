"""Form options and payload construction for the Streamlit app.

Allowed values come from the training schema (churn.data), so the form can
only offer categories the API accepts. The form asks about phone and internet
service first and fills dependent fields ("No phone service",
"No internet service") automatically, so users cannot submit inconsistent
combinations.
"""

from churn.data import CATEGORICAL_VALUES, INTERNET_ADDON_COLUMNS

INTERNET_ADDON_LABELS = {
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
}
assert list(INTERNET_ADDON_LABELS) == INTERNET_ADDON_COLUMNS

CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
INTERNET_OPTIONS = ["Fiber optic", "DSL", "No"]
PAYMENT_OPTIONS = sorted(CATEGORICAL_VALUES["PaymentMethod"])
GENDER_OPTIONS = sorted(CATEGORICAL_VALUES["gender"])
YES_NO = ["No", "Yes"]

assert set(CONTRACT_OPTIONS) == CATEGORICAL_VALUES["Contract"]
assert set(INTERNET_OPTIONS) == CATEGORICAL_VALUES["InternetService"]

DEFAULTS = {
    "gender": "Female",
    "senior_citizen": "No",
    "partner": "No",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "addons": {col: "No" for col in INTERNET_ADDON_COLUMNS},
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 70.0,
    "auto_total_charges": True,
    "total_charges": 840.0,
}


def estimated_total_charges(tenure: int, monthly_charges: float) -> float:
    """Convenience estimate for the form (tenure × monthly charge); users can override it."""
    return round(tenure * monthly_charges, 2)


def build_customer_payload(
    *,
    gender: str,
    senior_citizen: str,
    partner: str,
    dependents: str,
    tenure: int,
    phone_service: str,
    multiple_lines: str,
    internet_service: str,
    addons: dict[str, str],
    contract: str,
    paperless_billing: str,
    payment_method: str,
    monthly_charges: float,
    total_charges: float,
) -> dict:
    """Translate form answers into the exact JSON body POST /predict expects."""
    has_internet = internet_service != "No"
    return {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "MultipleLines": multiple_lines if phone_service == "Yes" else "No phone service",
        "InternetService": internet_service,
        **{col: (addons[col] if has_internet else "No internet service") for col in INTERNET_ADDON_COLUMNS},
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }
