"""Regression tests: every unusable artifact raises ModelArtifactError, and the
API reports it through /health instead of crashing.

Found in Phase 12: corrupted metadata JSON, missing metadata keys, and an
unreadable pickle previously raised JSONDecodeError / KeyError / ValueError.
"""

import hashlib
import json

import joblib
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from churn import data, predict, train

VERSION = "0.0.1-test"
LOCKED = {
    "model": "HistGradientBoosting",
    "hyperparameters": {"model__max_iter": 20, "model__max_leaf_nodes": 7},
    "threshold": 0.3,
}


@pytest.fixture
def artifact_dir(clean_df, tmp_path):
    X, y = data.split_features_target(clean_df)
    pipeline = train.fit_final_pipeline(X, y, LOCKED)
    train.save_artifact(pipeline, train.build_metadata(LOCKED, X, y, VERSION), train.artifact_paths(tmp_path, VERSION))
    return tmp_path


def _paths(directory):
    return train.artifact_paths(directory, VERSION)


def _rewrite_metadata(directory, change):
    meta_path = _paths(directory).metadata
    meta = json.loads(meta_path.read_text())
    change(meta)
    meta_path.write_text(json.dumps(meta))


def _replace_model_bytes(directory, payload: bytes):
    """Overwrite the model file and update the checksum so only the content is wrong."""
    _paths(directory).model.write_bytes(payload)
    _rewrite_metadata(directory, lambda m: m.update(artifact_sha256=hashlib.sha256(payload).hexdigest()))


def test_intact_artifact_loads(artifact_dir):
    assert predict.load_model(artifact_dir, VERSION).version == VERSION


def test_missing_metadata_file(artifact_dir):
    _paths(artifact_dir).metadata.unlink()
    with pytest.raises(predict.ModelArtifactError, match="Missing"):
        predict.load_model(artifact_dir, VERSION)


def test_invalid_metadata_json(artifact_dir):
    _paths(artifact_dir).metadata.write_text("{not json")
    with pytest.raises(predict.ModelArtifactError, match="not valid JSON"):
        predict.load_model(artifact_dir, VERSION)


@pytest.mark.parametrize("key", sorted(predict.REQUIRED_METADATA_KEYS))
def test_metadata_missing_required_key(artifact_dir, key):
    _rewrite_metadata(artifact_dir, lambda m: m.pop(key))
    with pytest.raises(predict.ModelArtifactError, match=key):
        predict.load_model(artifact_dir, VERSION)


def test_version_mismatch(artifact_dir):
    _rewrite_metadata(artifact_dir, lambda m: m.update(model_version="2.0.0"))
    with pytest.raises(predict.ModelArtifactError, match="version"):
        predict.load_model(artifact_dir, VERSION)


def test_unreadable_pickle_with_matching_checksum(artifact_dir):
    _replace_model_bytes(artifact_dir, b"not a pickle")
    with pytest.raises(predict.ModelArtifactError, match="Could not load"):
        predict.load_model(artifact_dir, VERSION)


def test_pickle_of_wrong_object(artifact_dir, tmp_path):
    other = tmp_path / "other.joblib"
    joblib.dump({"not": "a pipeline"}, other)
    _replace_model_bytes(artifact_dir, other.read_bytes())
    with pytest.raises(predict.ModelArtifactError, match="Pipeline"):
        predict.load_model(artifact_dir, VERSION)


def test_sklearn_version_mismatch_is_logged_not_fatal(artifact_dir, caplog):
    _rewrite_metadata(artifact_dir, lambda m: m["environment"].update({"scikit-learn": "0.0.0"}))
    with caplog.at_level("WARNING", logger="churn.predict"):
        predict.load_model(artifact_dir, VERSION)
    assert "0.0.0" in caplog.text


def test_api_reports_corrupted_artifact_as_unavailable(artifact_dir):
    _paths(artifact_dir).metadata.write_text("{not json")
    app = create_app(model_loader=lambda: predict.load_model(artifact_dir, VERSION))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json={}).status_code in (422, 503)
