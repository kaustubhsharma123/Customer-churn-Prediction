"""Minimal HTTP client for the churn API, used by the Streamlit app.

No model or preprocessing logic lives here: the client sends a customer
record to POST /predict and turns HTTP outcomes into clear exceptions.
"""

import os
from dataclasses import dataclass

import requests

DEFAULT_API_URL = os.environ.get("CHURN_API_URL", "http://127.0.0.1:8000")
DEFAULT_TIMEOUT_SECONDS = 10


class ApiError(Exception):
    """Unexpected API response."""


class ApiUnavailableError(ApiError):
    """API cannot be reached, timed out, or reports the model is not loaded."""


class ApiValidationError(ApiError):
    """API rejected the customer record (HTTP 422)."""

    def __init__(self, messages: list[str]):
        self.messages = messages
        super().__init__("; ".join(messages))


@dataclass(frozen=True)
class PredictionResult:
    churn_probability: float
    prediction: int
    risk_level: str
    threshold: float
    model_version: str


def format_validation_errors(detail) -> list[str]:
    """Readable messages from either 422 shape the API returns.

    - field-level errors: [{"loc": ["body", "tenure"], "msg": "...", "type": "..."}]
    - schema-rule errors: ["3 rows where internet add-ons disagree ..."]
    """
    if not isinstance(detail, list):
        return [str(detail)]
    messages = []
    for item in detail:
        if isinstance(item, dict):
            location = [str(part) for part in item.get("loc", []) if part != "body"]
            field = ".".join(location) or "request"
            messages.append(f"{field}: {item.get('msg', 'invalid value')}")
        else:
            messages.append(str(item))
    return messages


class ChurnApiClient:
    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = DEFAULT_TIMEOUT_SECONDS, session=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()  # injectable for tests

    def _request(self, method: str, path: str, **kwargs):
        try:
            return self._session.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise ApiUnavailableError(f"Cannot reach the API at {self.base_url} ({type(exc).__name__})") from exc

    def health(self) -> dict:
        """Health payload; raises ApiUnavailableError if the service or model is down."""
        response = self._request("GET", "/health")
        if response.status_code == 503:
            raise ApiUnavailableError("API is running but the model is not loaded")
        if response.status_code != 200:
            raise ApiError(f"Unexpected /health status {response.status_code}")
        return response.json()

    def predict(self, customer: dict) -> PredictionResult:
        response = self._request("POST", "/predict", json=customer)
        if response.status_code == 200:
            body = response.json()
            try:
                return PredictionResult(**{f: body[f] for f in PredictionResult.__dataclass_fields__})
            except KeyError as exc:
                raise ApiError(f"Prediction response is missing {exc}") from exc
        if response.status_code == 422:
            raise ApiValidationError(format_validation_errors(_json_or_text(response).get("detail")))
        if response.status_code == 503:
            raise ApiUnavailableError("API is running but the model is not loaded")
        raise ApiError(f"Unexpected /predict status {response.status_code}: {_json_or_text(response)}")


def _json_or_text(response) -> dict:
    try:
        body = response.json()
        return body if isinstance(body, dict) else {"detail": body}
    except ValueError:
        return {"detail": response.text[:200]}
