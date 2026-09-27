"""Inference: load the saved pipeline and score one customer.

Kept separate from training code. Preprocessing happens only inside the saved
Pipeline, so inference applies exactly what was fitted during training.
Inputs are validated against the same schema as the training data
(churn.data.validate_data) before prediction.
"""

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from churn.config import MODEL_VERSION, MODELS_DIR
from churn.data import DataValidationError, validate_data
from churn.preprocessing import NUMERIC_FEATURES, model_features

logger = logging.getLogger(__name__)


class InvalidCustomerError(ValueError):
    """Customer record failed validation; `errors` lists every problem found."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid customer record:\n- " + "\n- ".join(errors))


class ModelArtifactError(RuntimeError):
    """The saved model artifact is missing, corrupted, or inconsistent."""


@dataclass(frozen=True)
class ChurnModel:
    pipeline: Pipeline
    metadata: dict

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])


@dataclass(frozen=True)
class Prediction:
    churn_probability: float
    prediction: int  # 1 = flag as likely to churn (probability >= threshold)
    risk_level: str  # LOW / MEDIUM / HIGH
    threshold: float
    model_version: str

    def to_dict(self) -> dict:
        return asdict(self)


# --- Loading -------------------------------------------------------------------
def load_model(models_dir: Path = MODELS_DIR, version: str = MODEL_VERSION) -> ChurnModel:
    """Load a versioned artifact and verify it against its metadata."""
    stem = f"churn_pipeline_v{version}"
    model_path, metadata_path = models_dir / f"{stem}.joblib", models_dir / f"{stem}.json"
    for path in (model_path, metadata_path):
        if not path.exists():
            raise ModelArtifactError(f"Missing {path}. Train it with `python -m churn.train`.")

    metadata = json.loads(metadata_path.read_text())
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_hash != metadata.get("artifact_sha256"):
        raise ModelArtifactError(f"{model_path} does not match its metadata checksum")
    if metadata.get("model_version") != version:
        raise ModelArtifactError(f"Metadata version {metadata.get('model_version')!r} != {version!r}")

    trained_with = metadata["environment"]["scikit-learn"]
    if trained_with != sklearn.__version__:
        # Pickled sklearn objects are only guaranteed to load correctly with the same version.
        logger.warning("Model trained with scikit-learn %s, running %s", trained_with, sklearn.__version__)

    return ChurnModel(pipeline=joblib.load(model_path), metadata=metadata)


# --- Validation -------------------------------------------------------------
def validate_customer(record: dict) -> pd.DataFrame:
    """Check one raw customer record and return it as a one-row DataFrame.

    Types are checked here; categories, ranges, and consistency rules reuse
    the training-data validator so both follow one schema.
    """
    if not isinstance(record, dict):
        raise InvalidCustomerError([f"Expected a dict, got {type(record).__name__}"])

    features = model_features()
    errors = [f"Missing field: {f}" for f in features if f not in record]
    errors += [f"Unknown field: {f}" for f in record if f not in features]

    for field in NUMERIC_FEATURES:
        value = record.get(field)
        if field not in record:
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            errors.append(f"{field}: expected a finite number, got {value!r}")
    tenure = record.get("tenure")
    if isinstance(tenure, Real) and not isinstance(tenure, bool) and math.isfinite(tenure) and tenure != int(tenure):
        errors.append(f"tenure: expected whole months, got {tenure!r}")
    if isinstance(record.get("SeniorCitizen"), bool):
        errors.append("SeniorCitizen: expected 0 or 1, got a boolean")
    if errors:
        raise InvalidCustomerError(errors)

    row = pd.DataFrame([{f: record[f] for f in features}])
    # Schema check reuses validate_data, which also needs an ID and PhoneService.
    # PhoneService is fully determined by MultipleLines, so derive it for the check only.
    check = row.assign(
        customerID="inference",
        PhoneService=(row["MultipleLines"] != "No phone service").map({True: "Yes", False: "No"}),
    )
    try:
        validate_data(check, require_target=False)
    except DataValidationError as exc:
        problems = [line.lstrip("- ") for line in str(exc).splitlines()[1:]] or [str(exc)]
        raise InvalidCustomerError(problems) from exc
    return row


# --- Prediction -------------------------------------------------------------
def risk_level(probability: float, risk_levels: dict) -> str:
    """Map a probability to its band; bands are [lower, upper) except the top one."""
    for name, (lower, upper) in risk_levels.items():
        if lower <= probability < upper or (upper == 1.0 and probability == 1.0):
            return name
    raise ValueError(f"Probability {probability} outside all risk bands")


def predict_customer(model: ChurnModel, record: dict) -> Prediction:
    """Validate one customer record and return probability, decision, and risk level."""
    row = validate_customer(record)
    probability = float(model.pipeline.predict_proba(row)[0, 1])
    return Prediction(
        churn_probability=probability,
        prediction=int(probability >= model.threshold),
        risk_level=risk_level(probability, model.metadata["risk_levels"]),
        threshold=model.threshold,
        model_version=model.version,
    )
