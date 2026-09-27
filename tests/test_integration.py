"""End-to-end integration tests on the real dataset and production artifact.

Flow under test: customer record -> Streamlit API client -> FastAPI -> saved
.joblib pipeline -> prediction response. Skipped when the dataset or the
(git-ignored) artifact is absent: run `python -m churn.data` and
`python -m churn.train` first. These tests never fit on or score the test split.
"""

import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from app import api_client, inputs
from churn import data, predict, train
from churn.config import LOCKED_DECISION_FILE, MODEL_VERSION, MODELS_DIR, PROJECT_ROOT, RAW_DATA_FILE
from churn.preprocessing import model_features
from churn.tuning import make_tuned_pipeline

ARTIFACT = MODELS_DIR / f"churn_pipeline_v{MODEL_VERSION}.joblib"
needs_data = pytest.mark.skipif(not RAW_DATA_FILE.exists(), reason="dataset not downloaded")
needs_artifact = pytest.mark.skipif(
    not (RAW_DATA_FILE.exists() and ARTIFACT.exists()), reason="dataset or production artifact missing"
)

# Fingerprint of the held-out test split (Phase 4). If this changes, the test set changed.
TEST_SPLIT_SHA256 = "768d6e78618c6a04af5019a8eac32b0dcf679610df65a40fbabd64fc122ccd45"


def as_record(row: pd.Series) -> dict:
    """Plain-Python customer record (as a JSON client would send it)."""
    return {f: (row[f].item() if hasattr(row[f], "item") else row[f]) for f in model_features()}


class _InProcessSession:
    """Lets the Streamlit client talk to the FastAPI app in-process."""

    def __init__(self, test_client):
        self._test_client = test_client

    def request(self, method, url, timeout=None, **kwargs):
        return self._test_client.request(method, url, **kwargs)


@pytest.fixture(scope="module")
def train_df():
    train_part, _ = data.split_train_test(data.load_dataset())
    return train_part


@pytest.fixture(scope="module")
def production_model():
    return predict.load_model()


@pytest.fixture(scope="module")
def api(production_model):
    # The default app wiring (real load_model), exactly as `uvicorn api.main:app` runs it.
    with TestClient(create_app()) as client:
        yield client


@pytest.fixture(scope="module")
def sample(train_df):
    # Deterministic sample covering all contracts and internet types.
    return train_df.groupby(["Contract", "InternetService"], group_keys=False).head(4)


# --- Data split and artifact provenance ------------------------------------------
@needs_data
def test_held_out_split_is_unchanged():
    train_part, test_part = data.split_train_test(data.load_dataset())
    assert (len(train_part), len(test_part), int(test_part["Churn"].sum())) == (5634, 1409, 374)
    ids = ",".join(sorted(test_part["customerID"]))
    assert hashlib.sha256(ids.encode()).hexdigest() == TEST_SPLIT_SHA256


@needs_artifact
def test_artifact_matches_locked_decision(production_model):
    locked = train.load_locked_decision(LOCKED_DECISION_FILE)
    meta = production_model.metadata
    assert meta["model"] == locked["model"]
    assert meta["hyperparameters"] == locked["hyperparameters"]
    assert meta["threshold"] == locked["threshold"] == 0.3
    assert meta["training_data"]["rows"] == 5634
    assert meta["training_data"]["source_sha256"] == data.DATASET_SHA256
    assert meta["input_schema"]["features"] == model_features()


# --- Training vs. inference preprocessing ------------------------------------------
@needs_artifact
def test_saved_preprocessing_equals_training_preprocessing(production_model, train_df):
    X_train, _ = data.split_features_target(train_df)
    locked = train.load_locked_decision()
    refit = make_tuned_pipeline(locked["model"], locked["hyperparameters"]).named_steps["preprocess"].fit(X_train)
    saved = production_model.pipeline.named_steps["preprocess"]

    assert list(saved.get_feature_names_out()) == list(refit.get_feature_names_out())
    np.testing.assert_array_equal(saved.transform(X_train), refit.transform(X_train))


@needs_artifact
def test_artifact_is_reproducible_from_training_data(production_model, train_df, tmp_path):
    rebuilt = predict.load_model(train.train_and_save(models_dir=tmp_path).model.parent)
    X_train, _ = data.split_features_target(train_df)
    np.testing.assert_array_equal(
        rebuilt.pipeline.predict_proba(X_train), production_model.pipeline.predict_proba(X_train)
    )


# --- Customer -> FastAPI -> artifact ---------------------------------------------------
@needs_artifact
def test_api_health_reports_production_model(api):
    assert api.get("/health").json() == {"status": "ok", "model_loaded": True, "model_version": MODEL_VERSION}


@needs_artifact
def test_api_predictions_match_saved_pipeline(api, production_model, sample):
    X_sample, _ = data.split_features_target(sample)
    batch_proba = production_model.pipeline.predict_proba(X_sample)[:, 1]
    bands = production_model.metadata["risk_levels"]

    for (_, row), expected in zip(sample.iterrows(), batch_proba):
        body = api.post("/predict", json=as_record(row)).json()
        assert body["churn_probability"] == pytest.approx(expected, abs=1e-12)  # single row == batch
        assert body["prediction"] == int(expected >= 0.3)
        assert body["risk_level"] == predict.risk_level(expected, bands)
        assert body["model_version"] == MODEL_VERSION


@needs_artifact
def test_api_is_deterministic_across_calls_and_reloads(api, sample):
    record = as_record(sample.iloc[0])
    first = api.post("/predict", json=record).json()
    assert all(api.post("/predict", json=record).json() == first for _ in range(3))
    with TestClient(create_app()) as fresh:
        assert fresh.post("/predict", json=record).json() == first


# --- Streamlit client -> FastAPI -> artifact -------------------------------------------
@needs_artifact
def test_streamlit_client_path_matches_saved_pipeline(api, production_model):
    client = api_client.ChurnApiClient("http://testserver", session=_InProcessSession(api))
    form = {k: v for k, v in inputs.DEFAULTS.items() if k != "auto_total_charges"}
    payload = inputs.build_customer_payload(**form)

    result = client.predict(payload)
    expected = production_model.pipeline.predict_proba(pd.DataFrame([payload]))[0, 1]
    assert result.churn_probability == pytest.approx(expected, abs=1e-12)
    assert result.threshold == 0.3 and result.model_version == MODEL_VERSION


# --- Invalid input is rejected identically on every path ----------------------------
INVALID = [
    ("Contract", "Weekly"),
    ("Contract", "one year"),
    ("gender", 0),
    ("SeniorCitizen", True),
    ("SeniorCitizen", 2),
    ("tenure", -1),
    ("tenure", 1.5),
    ("tenure", "12"),
    ("MonthlyCharges", 0),
    ("MonthlyCharges", "70"),
    ("TotalCharges", -1.0),
    ("TotalCharges", None),
    ("OnlineSecurity", "No internet service"),  # contradicts InternetService="Fiber optic"
]


@needs_artifact
@pytest.mark.parametrize("field, value", INVALID)
def test_invalid_input_rejected_on_every_path(api, sample, field, value):
    record = as_record(sample[sample["InternetService"] == "Fiber optic"].iloc[0])
    record[field] = value

    with pytest.raises(predict.InvalidCustomerError):
        predict.validate_customer(dict(record))
    assert api.post("/predict", json=record).status_code == 422
    client = api_client.ChurnApiClient("http://testserver", session=_InProcessSession(api))
    with pytest.raises(api_client.ApiValidationError):
        client.predict(record)


@needs_artifact
@pytest.mark.parametrize("field, value", [("tenure", 12.0), ("SeniorCitizen", 1.0)])
def test_api_is_stricter_than_python_for_whole_number_floats(api, sample, field, value):
    """Known, intentional difference: the API requires JSON integers; the Python
    validator accepts numerically whole floats. Nothing invalid gets through either path."""
    record = as_record(sample.iloc[0])
    record[field] = value
    predict.validate_customer(dict(record))
    assert api.post("/predict", json=record).status_code == 422


# --- Threshold / risk bands at exact boundaries, through the API ------------------------
class _FixedProbabilityPipeline:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, X):
        return np.array([[1 - self.probability, self.probability]])


@needs_artifact
@pytest.mark.parametrize("probability, prediction, risk", [
    (0.0, 0, "LOW"),
    (0.2999999, 0, "LOW"),
    (0.3, 1, "MEDIUM"),
    (0.5999999, 1, "MEDIUM"),
    (0.6, 1, "HIGH"),
    (1.0, 1, "HIGH"),
])
def test_threshold_and_bands_at_boundaries(production_model, sample, probability, prediction, risk):
    stub = predict.ChurnModel(_FixedProbabilityPipeline(probability), production_model.metadata)
    with TestClient(create_app(model_loader=lambda: stub)) as client:
        body = client.post("/predict", json=as_record(sample.iloc[0])).json()
    assert (body["prediction"], body["risk_level"]) == (prediction, risk)
    assert body["churn_probability"] == probability


# --- API and UI never bypass the production inference path -----------------------------
def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("folder, forbidden", [
    ("api", {"joblib", "sklearn", "churn.preprocessing", "churn.train", "churn.tuning", "churn.modeling"}),
    ("app", {"joblib", "sklearn", "churn.predict", "churn.train", "churn.preprocessing", "churn.explain",
             "api", "api.main"}),
])
def test_serving_code_does_not_bypass_inference(folder, forbidden):
    for path in (PROJECT_ROOT / folder).glob("*.py"):
        used = {name for name in _imports(path) if name.split(".")[0] in {f.split(".")[0] for f in forbidden}}
        assert not (used & forbidden) and not any(u.startswith("sklearn") for u in used), (path.name, used)


def test_api_predict_goes_through_churn_predict(monkeypatch, clean_df):
    import api.main

    calls = []
    real = predict.predict_customer

    def spy(model, record):
        calls.append(record)
        return real(model, record)

    monkeypatch.setattr(api.main, "predict_customer", spy)
    X, y = data.split_features_target(clean_df)
    locked = {"model": "HistGradientBoosting", "hyperparameters": {"model__max_iter": 10}, "threshold": 0.3}
    model = predict.ChurnModel(train.fit_final_pipeline(X, y, locked),
                               train.build_metadata(locked, X, y, "0.0.1-test"))
    with TestClient(create_app(model_loader=lambda: model)) as client:
        assert client.post("/predict", json=as_record(clean_df.iloc[0])).status_code == 200
    assert len(calls) == 1
