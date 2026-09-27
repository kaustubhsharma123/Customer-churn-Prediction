"""API tests. A small model trained on synthetic data is injected, so these
tests do not depend on the (git-ignored) production artifact."""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.schemas import CustomerFeatures
from churn import data, predict, train
from churn.config import MODELS_DIR, MODEL_VERSION
from churn.preprocessing import model_features

LOCKED = {
    "model": "HistGradientBoosting",
    "hyperparameters": {"model__max_iter": 30, "model__max_leaf_nodes": 7},
    "threshold": 0.3,
}
VALID_CUSTOMER = CustomerFeatures.model_config["json_schema_extra"]["example"]


@pytest.fixture(scope="module")
def test_model(tmp_path_factory):
    from tests.conftest import make_clean_dataset

    X, y = data.split_features_target(make_clean_dataset())
    pipeline = train.fit_final_pipeline(X, y, LOCKED)
    metadata = train.build_metadata(LOCKED, X, y, version="0.0.1-test")
    models_dir = tmp_path_factory.mktemp("models")
    train.save_artifact(pipeline, metadata, train.artifact_paths(models_dir, "0.0.1-test"))
    return predict.load_model(models_dir, "0.0.1-test")


@pytest.fixture(scope="module")
def client(test_model):
    with TestClient(create_app(model_loader=lambda: test_model)) as test_client:
        yield test_client


@pytest.fixture
def customer():
    return dict(VALID_CUSTOMER)


# --- Schema -------------------------------------------------------------------
def test_request_schema_matches_model_features():
    assert set(CustomerFeatures.model_fields) == set(model_features())


def test_example_customer_is_valid_for_inference():
    predict.validate_customer(dict(VALID_CUSTOMER))


# --- Health / model -----------------------------------------------------------
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model_version": "0.0.1-test"}


def test_health_when_model_fails_to_load():
    def broken_loader():
        raise predict.ModelArtifactError("missing")

    with TestClient(create_app(model_loader=broken_loader)) as broken:
        health = broken.get("/health")
        assert health.status_code == 503
        assert health.json()["model_loaded"] is False
        assert broken.post("/predict", json=VALID_CUSTOMER).status_code == 503
        assert broken.get("/model").status_code == 503


def test_model_metadata(client):
    response = client.get("/model")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "0.0.1-test"
    assert body["threshold"] == 0.3
    assert body["features"] == model_features()
    assert set(body["risk_levels"]) == {"LOW", "MEDIUM", "HIGH"}


# --- Prediction ---------------------------------------------------------------
def test_valid_prediction_response_schema(client, customer):
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"churn_probability", "prediction", "risk_level", "threshold", "model_version"}
    assert 0 <= body["churn_probability"] <= 1
    assert body["prediction"] in (0, 1)
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert body["prediction"] == int(body["churn_probability"] >= body["threshold"])


def test_api_matches_direct_inference(client, test_model, customer):
    api_result = client.post("/predict", json=customer).json()
    direct = predict.predict_customer(test_model, customer).to_dict()
    assert api_result == pytest.approx(direct)


def test_prediction_is_deterministic(client, customer):
    assert client.post("/predict", json=customer).json() == client.post("/predict", json=customer).json()


def test_boundary_values_accepted(client, customer):
    customer.update(tenure=0, TotalCharges=0.0, MonthlyCharges=0.01, SeniorCitizen=1)
    assert client.post("/predict", json=customer).status_code == 200


def test_integer_charges_accepted(client, customer):
    customer.update(MonthlyCharges=70, TotalCharges=140)
    assert client.post("/predict", json=customer).status_code == 200


# --- Invalid input ------------------------------------------------------------
def test_missing_field_returns_422(client, customer):
    del customer["Contract"]
    response = client.post("/predict", json=customer)
    assert response.status_code == 422
    assert any(err["loc"][-1] == "Contract" for err in response.json()["detail"])


def test_unknown_field_returns_422(client, customer):
    customer["customerID"] = "1234-ABCDE"
    assert client.post("/predict", json=customer).status_code == 422


def test_empty_body_returns_422(client):
    assert client.post("/predict", json={}).status_code == 422


@pytest.mark.parametrize("field, value", [
    ("Contract", "Weekly"),
    ("Contract", "month-to-month"),  # case-sensitive
    ("InternetService", ""),
    ("PaymentMethod", None),
    ("gender", 1),
    ("SeniorCitizen", 2),
    ("SeniorCitizen", -1),
    ("SeniorCitizen", True),
    ("SeniorCitizen", "1"),
])
def test_invalid_categorical_values_return_422(client, customer, field, value):
    customer[field] = value
    assert client.post("/predict", json=customer).status_code == 422


@pytest.mark.parametrize("field, value", [
    ("tenure", -1),
    ("tenure", 12.5),
    ("tenure", "12"),
    ("MonthlyCharges", 0),
    ("MonthlyCharges", -5.0),
    ("MonthlyCharges", "70.7"),
    ("TotalCharges", -0.01),
])
def test_invalid_numeric_values_return_422(client, customer, field, value):
    customer[field] = value
    assert client.post("/predict", json=customer).status_code == 422


def test_non_finite_number_returns_422(client, customer):
    body = __import__("json").dumps(customer).replace('"MonthlyCharges": 70.7', '"MonthlyCharges": NaN')
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "MonthlyCharges"


def test_validation_errors_do_not_echo_input(client, customer):
    customer["Contract"] = "Weekly"
    detail = client.post("/predict", json=customer).json()["detail"]
    assert all(set(err) == {"loc", "msg", "type"} for err in detail)


def test_inconsistent_services_rejected_by_schema_rules(client, customer):
    # Each field is individually valid; the combination violates the training-data rules.
    customer.update(InternetService="No", OnlineSecurity="Yes")
    response = client.post("/predict", json=customer)
    assert response.status_code == 422
    assert any("InternetService" in msg for msg in response.json()["detail"])


def test_malformed_json_returns_422(client):
    response = client.post("/predict", content="{not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422


# --- Real artifact (integration) ----------------------------------------------
@pytest.mark.skipif(
    not (MODELS_DIR / f"churn_pipeline_v{MODEL_VERSION}.joblib").exists(),
    reason="production artifact not built (python -m churn.train)",
)
def test_default_app_loads_production_artifact():
    from api.main import app

    with TestClient(app) as prod_client:
        health = prod_client.get("/health").json()
        assert health == {"status": "ok", "model_loaded": True, "model_version": MODEL_VERSION}
        assert prod_client.post("/predict", json=VALID_CUSTOMER).status_code == 200
