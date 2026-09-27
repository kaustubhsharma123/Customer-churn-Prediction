"""Tests for the production artifact (train.py) and inference (predict.py)."""

import json

import joblib
import numpy as np
import pytest

from churn import data, predict, train
from churn.config import LOCKED_DECISION_FILE, RAW_DATA_FILE
from churn.preprocessing import model_features

LOCKED = {
    "model": "HistGradientBoosting",
    "hyperparameters": {"model__max_iter": 30, "model__max_leaf_nodes": 7},
    "threshold": 0.3,
}


@pytest.fixture
def artifact(clean_df, tmp_path):
    X, y = data.split_features_target(clean_df)
    pipeline = train.fit_final_pipeline(X, y, LOCKED)
    metadata = train.build_metadata(LOCKED, X, y, version="0.0.1-test")
    paths = train.save_artifact(pipeline, metadata, train.artifact_paths(tmp_path, "0.0.1-test"))
    return paths, predict.load_model(tmp_path, "0.0.1-test"), X


@pytest.fixture
def customer(clean_df):
    record = clean_df.iloc[0][model_features()].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()}  # plain Python types


# --- Artifact creation / loading ---------------------------------------------
def test_artifact_files_and_metadata(artifact):
    paths, model, X = artifact
    assert paths.model.exists() and paths.metadata.exists()
    meta = json.loads(paths.metadata.read_text())
    assert meta["model_version"] == "0.0.1-test"
    assert meta["threshold"] == 0.3
    assert meta["input_schema"]["features"] == model_features()
    assert meta["training_data"]["rows"] == len(X)
    assert meta["training_data"]["test_set_used_for_fitting"] is False
    assert list(meta["risk_levels"]) == ["LOW", "MEDIUM", "HIGH"]
    assert len(meta["artifact_sha256"]) == 64


def test_loaded_pipeline_contains_preprocessing_and_model(artifact):
    _, model, X = artifact
    assert list(model.pipeline.named_steps) == ["preprocess", "model"]
    assert model.pipeline.predict_proba(X.head(3)).shape == (3, 2)


def test_missing_artifact_raises(tmp_path):
    with pytest.raises(predict.ModelArtifactError, match="Missing"):
        predict.load_model(tmp_path, "9.9.9")


def test_tampered_artifact_is_rejected(artifact, tmp_path):
    paths, _, _ = artifact
    joblib.dump({"not": "a model"}, paths.model)
    with pytest.raises(predict.ModelArtifactError, match="checksum"):
        predict.load_model(tmp_path, "0.0.1-test")


def test_locked_decision_file_is_complete():
    locked = train.load_locked_decision(LOCKED_DECISION_FILE)
    assert locked["model"] == "HistGradientBoosting"
    assert locked["threshold"] == 0.3


# --- Prediction ----------------------------------------------------------------
def test_valid_prediction_structure_and_range(artifact, customer):
    _, model, _ = artifact
    result = predict.predict_customer(model, customer)
    assert 0.0 <= result.churn_probability <= 1.0
    assert result.prediction in (0, 1)
    assert result.risk_level in ("LOW", "MEDIUM", "HIGH")
    assert result.model_version == "0.0.1-test"
    assert set(result.to_dict()) == {"churn_probability", "prediction", "risk_level", "threshold", "model_version"}


def test_inference_matches_pipeline_on_training_rows(artifact):
    _, model, X = artifact
    for i in range(5):
        record = {k: (v.item() if hasattr(v, "item") else v) for k, v in X.iloc[i][model_features()].items()}
        expected = model.pipeline.predict_proba(X.iloc[[i]])[0, 1]
        assert predict.predict_customer(model, record).churn_probability == pytest.approx(expected)


def test_prediction_is_deterministic(artifact, customer):
    _, model, _ = artifact
    first = predict.predict_customer(model, customer)
    second = predict.predict_customer(model, dict(customer))
    assert first == second


def test_extra_phone_service_is_not_required(artifact, customer):
    _, model, _ = artifact
    assert "PhoneService" not in customer
    predict.predict_customer(model, customer)


# --- Threshold and risk levels -----------------------------------------------
@pytest.mark.parametrize(
    "probability, expected",
    [(0.0, "LOW"), (0.2999, "LOW"), (0.3, "MEDIUM"), (0.5999, "MEDIUM"), (0.6, "HIGH"), (1.0, "HIGH")],
)
def test_risk_level_boundaries(probability, expected):
    bands = {"LOW": [0.0, 0.3], "MEDIUM": [0.3, 0.6], "HIGH": [0.6, 1.0]}
    assert predict.risk_level(probability, bands) == expected


def test_prediction_flag_follows_threshold(artifact, customer):
    _, model, _ = artifact
    probability = predict.predict_customer(model, customer).churn_probability

    for threshold, expected in [(probability - 1e-9, 1), (probability + 1e-9, 0)]:
        adjusted = predict.ChurnModel(model.pipeline, {**model.metadata, "threshold": threshold})
        assert predict.predict_customer(adjusted, customer).prediction == expected


def test_flag_and_risk_level_agree(artifact, clean_df):
    _, model, _ = artifact
    for i in range(20):
        record = {k: (v.item() if hasattr(v, "item") else v) for k, v in clean_df.iloc[i][model_features()].items()}
        result = predict.predict_customer(model, record)
        assert (result.prediction == 1) == (result.risk_level != "LOW")


# --- Invalid input -------------------------------------------------------------
def test_missing_field_rejected(customer):
    del customer["Contract"]
    with pytest.raises(predict.InvalidCustomerError, match="Missing field: Contract"):
        predict.validate_customer(customer)


def test_unknown_field_rejected(customer):
    customer["Contrct"] = "One year"
    with pytest.raises(predict.InvalidCustomerError, match="Unknown field: Contrct"):
        predict.validate_customer(customer)


@pytest.mark.parametrize("field, value", [
    ("Contract", "Weekly"),
    ("tenure", -1),
    ("tenure", 12.5),
    ("tenure", "12"),
    ("MonthlyCharges", 0),
    ("MonthlyCharges", float("nan")),
    ("TotalCharges", None),
    ("SeniorCitizen", True),
    ("SeniorCitizen", "1"),
])
def test_invalid_values_rejected(customer, field, value):
    customer[field] = value
    with pytest.raises(predict.InvalidCustomerError):
        predict.validate_customer(customer)


def test_inconsistent_internet_services_rejected(customer):
    customer["InternetService"] = "No"
    customer["OnlineSecurity"] = "Yes"
    with pytest.raises(predict.InvalidCustomerError, match="InternetService"):
        predict.validate_customer(customer)


def test_all_errors_reported_together(customer):
    del customer["tenure"]
    customer["gender"] = "Unknown"
    customer["extra"] = 1
    with pytest.raises(predict.InvalidCustomerError) as exc_info:
        predict.validate_customer(customer)
    assert len(exc_info.value.errors) == 2  # type/field problems reported before schema checks
    assert any("tenure" in e for e in exc_info.value.errors)
    assert any("extra" in e for e in exc_info.value.errors)


def test_non_dict_rejected():
    with pytest.raises(predict.InvalidCustomerError):
        predict.validate_customer(["not", "a", "dict"])


# --- Real training split (integration) -------------------------------------
@pytest.mark.skipif(not RAW_DATA_FILE.exists(), reason="raw dataset not downloaded")
def test_train_and_save_uses_training_split_only(tmp_path):
    paths = train.train_and_save(models_dir=tmp_path, version="0.0.2-test")
    meta = json.loads(paths.metadata.read_text())
    train_df, test_df = data.split_train_test(data.load_dataset())
    assert meta["training_data"]["rows"] == len(train_df) == 5634
    assert meta["hyperparameters"] == train.load_locked_decision()["hyperparameters"]

    model = predict.load_model(tmp_path, "0.0.2-test")
    X_train, _ = data.split_features_target(train_df)
    proba = model.pipeline.predict_proba(X_train)[:, 1]
    assert np.isfinite(proba).all()
