"""Request/response models for the churn API.

Allowed categories are generated from the training schema
(churn.data.CATEGORICAL_VALUES), so the API and the model cannot drift apart.
Pydantic handles field-level checks and documentation; churn.predict then
applies the full schema validation (including cross-field consistency).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from churn.data import CATEGORICAL_VALUES


def _choices(column: str):
    """Literal type of the allowed values for a categorical column."""
    return Literal[tuple(sorted(CATEGORICAL_VALUES[column]))]


class CustomerFeatures(BaseModel):
    """One customer's attributes (all fields required; unknown fields rejected)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
                "tenure": 2, "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
                "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 70.7, "TotalCharges": 151.65,
            }
        },
    )

    tenure: int = Field(strict=True, ge=0, description="Months with the company (whole months)")
    MonthlyCharges: float = Field(strict=True, gt=0, allow_inf_nan=False, description="Current monthly charge")
    TotalCharges: float = Field(strict=True, ge=0, allow_inf_nan=False, description="Total billed to date (0 if not yet billed)")
    gender: _choices("gender")
    SeniorCitizen: int = Field(strict=True, ge=0, le=1, description="1 = senior citizen, 0 = not")
    Partner: _choices("Partner")
    Dependents: _choices("Dependents")
    PaperlessBilling: _choices("PaperlessBilling")
    MultipleLines: _choices("MultipleLines")
    InternetService: _choices("InternetService")
    OnlineSecurity: _choices("OnlineSecurity")
    OnlineBackup: _choices("OnlineBackup")
    DeviceProtection: _choices("DeviceProtection")
    TechSupport: _choices("TechSupport")
    StreamingTV: _choices("StreamingTV")
    StreamingMovies: _choices("StreamingMovies")
    Contract: _choices("Contract")
    PaymentMethod: _choices("PaymentMethod")


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    prediction: Literal[0, 1] = Field(description="1 = flagged as likely to churn (probability >= threshold)")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_version: str
    model: str
    created_at_utc: str
    threshold: float
    threshold_rule: str | None
    risk_levels: dict[str, list[float]]
    features: list[str]
    training_rows: int
    cv_metrics_at_threshold: dict[str, float] | None
