"""Tests for the Streamlit app's helpers and API client (no browser tests).

The client is exercised against the real FastAPI app through its TestClient
(a requests-compatible session), so the frontend/backend contract is tested
without a network or the production artifact.
"""

import itertools

import pytest
import requests
from fastapi.testclient import TestClient

from api.main import create_app
from app import api_client, inputs
from churn import data, predict, train
from churn.config import PROJECT_ROOT
from churn.preprocessing import model_features

LOCKED = {
    "model": "HistGradientBoosting",
    "hyperparameters": {"model__max_iter": 30, "model__max_leaf_nodes": 7},
    "threshold": 0.3,
}


def default_form(**overrides) -> dict:
    form = {k: v for k, v in inputs.DEFAULTS.items() if k != "auto_total_charges"}
    form["addons"] = dict(form["addons"])
    form.update(overrides)
    return form


@pytest.fixture(scope="module")
def api_session(tmp_path_factory):
    from tests.conftest import make_clean_dataset

    X, y = data.split_features_target(make_clean_dataset())
    models_dir = tmp_path_factory.mktemp("models")
    pipeline = train.fit_final_pipeline(X, y, LOCKED)
    train.save_artifact(pipeline, train.build_metadata(LOCKED, X, y, "0.0.1-test"),
                        train.artifact_paths(models_dir, "0.0.1-test"))
    model = predict.load_model(models_dir, "0.0.1-test")
    with TestClient(create_app(model_loader=lambda: model)) as session:
        yield _InProcessSession(session)


class _InProcessSession:
    """Adapts TestClient to the client's session interface. The timeout that a real
    requests.Session needs is meaningless in-process, so it is dropped here."""

    def __init__(self, test_client):
        self._test_client = test_client

    def request(self, method, url, timeout=None, **kwargs):
        return self._test_client.request(method, url, **kwargs)


@pytest.fixture
def client(api_session):
    return api_client.ChurnApiClient("http://testserver", session=api_session)


# --- Form payloads -----------------------------------------------------------------
def test_default_payload_has_exactly_the_api_fields():
    payload = inputs.build_customer_payload(**default_form())
    assert set(payload) == set(model_features())


@pytest.mark.parametrize(
    "phone, internet, senior",
    list(itertools.product(["Yes", "No"], inputs.INTERNET_OPTIONS, ["Yes", "No"])),
)
def test_every_form_combination_passes_schema_validation(phone, internet, senior):
    payload = inputs.build_customer_payload(
        **default_form(phone_service=phone, internet_service=internet, senior_citizen=senior)
    )
    predict.validate_customer(payload)  # raises if the UI could ever send an invalid record


def test_no_internet_fills_addons_consistently():
    payload = inputs.build_customer_payload(**default_form(internet_service="No", addons={}))
    assert all(payload[col] == "No internet service" for col in inputs.INTERNET_ADDON_LABELS)


def test_no_phone_sets_multiple_lines():
    payload = inputs.build_customer_payload(**default_form(phone_service="No", multiple_lines="Yes"))
    assert payload["MultipleLines"] == "No phone service"


def test_senior_citizen_and_numeric_types():
    payload = inputs.build_customer_payload(**default_form(senior_citizen="Yes", tenure=5.0))
    assert payload["SeniorCitizen"] == 1
    assert isinstance(payload["tenure"], int)
    assert isinstance(payload["MonthlyCharges"], float)


def test_estimated_total_charges():
    assert inputs.estimated_total_charges(12, 70.0) == 840.0
    assert inputs.estimated_total_charges(0, 70.0) == 0.0


def test_form_options_match_schema():
    for col in ("Contract", "InternetService", "PaymentMethod", "gender"):
        options = {"Contract": inputs.CONTRACT_OPTIONS, "InternetService": inputs.INTERNET_OPTIONS,
                   "PaymentMethod": inputs.PAYMENT_OPTIONS, "gender": inputs.GENDER_OPTIONS}[col]
        assert set(options) == data.CATEGORICAL_VALUES[col]


# --- API client against the real FastAPI app ---------------------------------------
def test_health(client):
    assert client.health()["model_loaded"] is True


def test_predict_returns_result(client):
    result = client.predict(inputs.build_customer_payload(**default_form()))
    assert 0 <= result.churn_probability <= 1
    assert result.risk_level in {"LOW", "MEDIUM", "HIGH"}
    assert result.prediction == int(result.churn_probability >= result.threshold)
    assert result.model_version == "0.0.1-test"


def test_field_validation_error_is_readable(client):
    payload = inputs.build_customer_payload(**default_form())
    payload["Contract"] = "Weekly"
    with pytest.raises(api_client.ApiValidationError) as exc_info:
        client.predict(payload)
    assert any(msg.startswith("Contract:") for msg in exc_info.value.messages)


def test_schema_rule_error_is_readable(client):
    payload = inputs.build_customer_payload(**default_form())
    payload.update(InternetService="No", OnlineSecurity="Yes")
    with pytest.raises(api_client.ApiValidationError) as exc_info:
        client.predict(payload)
    assert any("InternetService" in msg for msg in exc_info.value.messages)


def test_model_not_loaded_is_unavailable():
    def broken():
        raise predict.ModelArtifactError("missing")

    with TestClient(create_app(model_loader=broken)) as session:
        broken_client = api_client.ChurnApiClient("http://testserver", session=_InProcessSession(session))
        with pytest.raises(api_client.ApiUnavailableError):
            broken_client.health()
        with pytest.raises(api_client.ApiUnavailableError):
            broken_client.predict(inputs.build_customer_payload(**default_form()))


# --- API client error handling (no server) ------------------------------------------
class _FakeResponse:
    def __init__(self, status_code, body=None, text=""):
        self.status_code, self._body, self.text = status_code, body, text

    def json(self):
        if self._body is None:
            raise ValueError("no JSON")
        return self._body


class _FakeSession:
    def __init__(self, response=None, error=None):
        self.response, self.error = response, error

    def request(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.response


def test_connection_error_is_unavailable():
    fake = api_client.ChurnApiClient("http://nowhere", session=_FakeSession(error=requests.ConnectionError()))
    with pytest.raises(api_client.ApiUnavailableError, match="Cannot reach"):
        fake.predict({})


def test_timeout_is_unavailable():
    fake = api_client.ChurnApiClient("http://slow", session=_FakeSession(error=requests.Timeout()))
    with pytest.raises(api_client.ApiUnavailableError):
        fake.health()


def test_server_error_is_reported():
    fake = api_client.ChurnApiClient("http://x", session=_FakeSession(_FakeResponse(500, text="boom")))
    with pytest.raises(api_client.ApiError, match="500"):
        fake.predict({})


def test_incomplete_success_response_is_rejected():
    fake = api_client.ChurnApiClient("http://x", session=_FakeSession(_FakeResponse(200, {"churn_probability": 0.4})))
    with pytest.raises(api_client.ApiError, match="missing"):
        fake.predict({})


def test_base_url_trailing_slash_is_normalised():
    assert api_client.ChurnApiClient("http://127.0.0.1:8000/").base_url == "http://127.0.0.1:8000"


@pytest.mark.parametrize("detail, expected", [
    ([{"loc": ["body", "tenure"], "msg": "Input should be a valid integer", "type": "int_type"}],
     ["tenure: Input should be a valid integer"]),
    (["1 rows where internet add-ons disagree with InternetService"],
     ["1 rows where internet add-ons disagree with InternetService"]),
    ("Model is not loaded", ["Model is not loaded"]),
])
def test_format_validation_errors(detail, expected):
    assert api_client.format_validation_errors(detail) == expected


# --- Streamlit script smoke test (headless, no browser) -------------------------------
def test_streamlit_app_renders_and_handles_unreachable_api(monkeypatch):
    from streamlit.testing.v1 import AppTest

    unreachable = "http://127.0.0.1:9"  # nothing listens on port 9
    monkeypatch.setenv("CHURN_API_URL", unreachable)
    script = PROJECT_ROOT / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(script), default_timeout=30).run()
    app.sidebar.text_input[0].set_value(unreachable).run()
    assert not app.exception
    assert app.title[0].value == "Customer churn risk"
    assert any("API not available" in w.value for w in app.sidebar.warning)

    app.button[0].click().run()
    assert not app.exception
    assert any("Cannot reach the API" in e.value for e in app.error)
