"""Train the production pipeline and save it as a versioned artifact.

Uses the locked Phase 7 decision (model, hyperparameters, threshold) and the
training split only; the held-out test set is never loaded for fitting.

    python -m churn.train      # train, save to models/, and run an inference smoke check

Artifacts (in models/):
    churn_pipeline_v<version>.joblib   fitted preprocessing + model Pipeline
    churn_pipeline_v<version>.json     metadata: threshold, risk bands, input schema,
                                       training data, library versions, file checksum
"""

import hashlib
import json
import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from churn.config import LOCKED_DECISION_FILE, MODEL_VERSION, MODELS_DIR, RANDOM_STATE, RAW_DATA_FILE
from churn.data import (
    CATEGORICAL_VALUES,
    DATASET_SHA256,
    load_dataset,
    split_features_target,
    split_train_test,
)
from churn.preprocessing import NUMERIC_FEATURES, model_features
from churn.tuning import make_tuned_pipeline

logger = logging.getLogger(__name__)

# Presentation bands on top of the decision threshold. LOW = below the action
# threshold (not flagged). HIGH starts where out-of-fold precision reaches ~0.71
# (~14% of training customers); see reports/threshold_analysis_tuned.csv.
HIGH_RISK_FROM = 0.60


@dataclass(frozen=True)
class ArtifactPaths:
    model: Path
    metadata: Path


def artifact_paths(models_dir: Path = MODELS_DIR, version: str = MODEL_VERSION) -> ArtifactPaths:
    stem = f"churn_pipeline_v{version}"
    return ArtifactPaths(model=models_dir / f"{stem}.joblib", metadata=models_dir / f"{stem}.json")


def load_locked_decision(path: Path = LOCKED_DECISION_FILE) -> dict:
    """The locked training/CV decision: model, hyperparameters, and threshold."""
    locked = json.loads(Path(path).read_text())
    missing = {"model", "hyperparameters", "threshold"} - set(locked)
    if missing:
        raise ValueError(f"Locked decision file {path} is missing {sorted(missing)}")
    return locked


def fit_final_pipeline(X_train: pd.DataFrame, y_train: pd.Series, locked: dict) -> Pipeline:
    """Fit the locked configuration (preprocessing + model in one Pipeline)."""
    return make_tuned_pipeline(locked["model"], locked["hyperparameters"]).fit(X_train, y_train)


def build_metadata(locked: dict, X_train: pd.DataFrame, y_train: pd.Series, version: str) -> dict:
    threshold = float(locked["threshold"])
    if not 0 < threshold < HIGH_RISK_FROM:
        raise ValueError(f"Threshold {threshold} must be between 0 and {HIGH_RISK_FROM}")
    return {
        "model_version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": locked["model"],
        "hyperparameters": locked["hyperparameters"],
        "threshold": threshold,
        "threshold_rule": locked.get("threshold_rule"),
        "risk_levels": {"LOW": [0.0, threshold], "MEDIUM": [threshold, HIGH_RISK_FROM], "HIGH": [HIGH_RISK_FROM, 1.0]},
        "input_schema": {
            "features": model_features(),
            "numeric": NUMERIC_FEATURES,
            "categorical": {
                col: sorted(CATEGORICAL_VALUES[col])
                for col in model_features() if col in CATEGORICAL_VALUES
            },
        },
        "training_data": {
            "source_sha256": DATASET_SHA256,
            "split": {"test_size": 0.2, "stratified": True, "random_state": RANDOM_STATE},
            "rows": int(len(X_train)),
            "churn_rate": round(float(y_train.mean()), 4),
            "test_set_used_for_fitting": False,
        },
        "cv_metrics_at_threshold": locked.get("cv_metrics_at_threshold"),
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }


def save_artifact(pipeline: Pipeline, metadata: dict, paths: ArtifactPaths) -> ArtifactPaths:
    """Write the pipeline, then the metadata including the pipeline file's checksum."""
    paths.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, paths.model)
    metadata = {**metadata, "artifact_sha256": hashlib.sha256(paths.model.read_bytes()).hexdigest()}
    paths.metadata.write_text(json.dumps(metadata, indent=2))
    return paths


def train_and_save(
    models_dir: Path = MODELS_DIR,
    version: str = MODEL_VERSION,
    locked_path: Path = LOCKED_DECISION_FILE,
    data_path: Path = RAW_DATA_FILE,
) -> ArtifactPaths:
    """Train the locked configuration on the training split and save the artifact."""
    locked = load_locked_decision(locked_path)
    train_df, _ = split_train_test(load_dataset(data_path))  # test half is discarded
    X_train, y_train = split_features_target(train_df)

    pipeline = fit_final_pipeline(X_train, y_train, locked)
    metadata = build_metadata(locked, X_train, y_train, version)
    paths = save_artifact(pipeline, metadata, artifact_paths(models_dir, version))
    logger.info("Saved %s (%d training rows)", paths.model, len(X_train))
    return paths


if __name__ == "__main__":
    from churn.predict import load_model, predict_customer

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    saved = train_and_save()
    print(f"Model:    {saved.model}\nMetadata: {saved.metadata}")

    # Smoke check on one *training* customer: load from disk and predict.
    model = load_model()
    train_df, _ = split_train_test(load_dataset())
    sample = train_df.iloc[0][model_features()].to_dict()
    result = predict_customer(model, sample)
    print(f"Smoke check (training customer {train_df.iloc[0]['customerID']}): {result.to_dict()}")
