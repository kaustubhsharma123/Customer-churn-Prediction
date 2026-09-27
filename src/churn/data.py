"""Load, clean, and validate the IBM Telco Customer Churn dataset.

Pipeline:  download_raw_data -> load_raw_data -> clean_data -> validate_data

Only deterministic, source-format fixes happen here. Anything learned from
the data (imputation statistics, scaling, encoding) belongs in the sklearn
preprocessing pipeline so it is fitted on training data only.

Run as a script to download (if needed), validate, and summarise the data:
    python -m churn.data
"""

import hashlib
import logging
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import RANDOM_STATE, RAW_DATA_FILE

logger = logging.getLogger(__name__)

# --- Source --------------------------------------------------------------
# IBM's repository (Apache-2.0). Checksum pins the exact file we validated.
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATASET_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

# --- Schema --------------------------------------------------------------
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

_YES_NO = {"Yes", "No"}
_INTERNET_ADDON = {"Yes", "No", "No internet service"}

# Allowed values per categorical feature, as observed in the source data.
# Unknown categories are rejected rather than silently passed downstream.
CATEGORICAL_VALUES: dict[str, set] = {
    "gender": {"Male", "Female"},
    "SeniorCitizen": {0, 1},
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": _INTERNET_ADDON,
    "OnlineBackup": _INTERNET_ADDON,
    "DeviceProtection": _INTERNET_ADDON,
    "TechSupport": _INTERNET_ADDON,
    "StreamingTV": _INTERNET_ADDON,
    "StreamingMovies": _INTERNET_ADDON,
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}
CATEGORICAL_COLUMNS = list(CATEGORICAL_VALUES)
INTERNET_ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
EXPECTED_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]


class DataValidationError(ValueError):
    """Raised when the dataset violates the expected schema or invariants."""


# --- Acquisition ---------------------------------------------------------
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_raw_data(path: Path = RAW_DATA_FILE, force: bool = False) -> Path:
    """Download the raw CSV (if absent) and verify its SHA-256 checksum."""
    if force or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading dataset from %s", DATASET_URL)
        urllib.request.urlretrieve(DATASET_URL, path)

    actual = _sha256(path)
    if actual != DATASET_SHA256:
        raise DataValidationError(
            f"Checksum mismatch for {path}: expected {DATASET_SHA256}, got {actual}. "
            "The source file may have changed; re-validate before using it."
        )
    return path


def load_raw_data(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Read the raw CSV exactly as stored, without any transformation."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run `python -m churn.data` to download it."
        )
    return pd.read_csv(path)


# --- Cleaning ------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known source-format issues. Returns a new DataFrame.

    - TotalCharges is stored as text; blank values occur only for customers
      with tenure == 0 (not yet billed), so they are set to 0.0. Any other
      non-numeric value is an error, not something to guess.
    - Churn "Yes"/"No" is mapped to 1/0.
    """
    df = df.copy()

    total_text = df["TotalCharges"].astype(str).str.strip()
    total = pd.to_numeric(total_text, errors="coerce")
    unparseable = total.isna()

    not_yet_billed = unparseable & (total_text == "") & (df["tenure"] == 0)
    unexpected = unparseable & ~not_yet_billed
    if unexpected.any():
        bad = df.loc[unexpected, [ID_COLUMN, "tenure", "TotalCharges"]]
        raise DataValidationError(
            f"{int(unexpected.sum())} TotalCharges values could not be parsed "
            f"and are not explained by tenure == 0:\n{bad.head(10).to_string()}"
        )
    if not_yet_billed.any():
        logger.info(
            "Set TotalCharges=0.0 for %d customers with tenure == 0",
            int(not_yet_billed.sum()),
        )
    df["TotalCharges"] = total.fillna(0.0).astype(float)

    if TARGET_COLUMN in df.columns:
        target_map = {"Yes": 1, "No": 0}
        mapped = df[TARGET_COLUMN].map(target_map)
        if mapped.isna().any():
            bad_values = sorted(df.loc[mapped.isna(), TARGET_COLUMN].astype(str).unique())
            raise DataValidationError(f"Unexpected {TARGET_COLUMN} values: {bad_values}")
        df[TARGET_COLUMN] = mapped.astype(int)

    return df


# --- Validation ----------------------------------------------------------
def validate_data(df: pd.DataFrame, require_target: bool = True) -> None:
    """Check schema and business invariants of a *cleaned* DataFrame.

    Collects every problem and raises a single DataValidationError listing
    all of them, so issues can be fixed in one pass.
    """
    required = [ID_COLUMN, *FEATURE_COLUMNS] + ([TARGET_COLUMN] if require_target else [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        # Later checks depend on these columns, so stop here.
        raise DataValidationError(f"Missing required columns: {missing}")

    problems: list[str] = []

    if df.empty:
        problems.append("Dataset is empty")

    null_counts = df[required].isna().sum()
    for col, count in null_counts[null_counts > 0].items():
        problems.append(f"{col}: {count} missing values")

    duplicate_ids = df[ID_COLUMN].duplicated().sum()
    if duplicate_ids:
        problems.append(f"{ID_COLUMN}: {duplicate_ids} duplicate IDs")

    for col, allowed in CATEGORICAL_VALUES.items():
        unknown = set(df[col].dropna().unique()) - allowed
        if unknown:
            problems.append(f"{col}: unexpected values {sorted(map(str, unknown))}")

    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"{col}: expected numeric dtype, got {df[col].dtype}")
            continue
        if (df[col] < 0).any():
            problems.append(f"{col}: {int((df[col] < 0).sum())} negative values")
    if pd.api.types.is_numeric_dtype(df["MonthlyCharges"]) and (df["MonthlyCharges"] == 0).any():
        problems.append("MonthlyCharges: zero values found")

    if require_target and not set(df[TARGET_COLUMN].unique()) <= {0, 1}:
        problems.append(f"{TARGET_COLUMN}: expected only 0/1 after cleaning")

    # Consistency: "No internet service" iff InternetService == "No".
    no_internet = df["InternetService"] == "No"
    addon_says_none = df[INTERNET_ADDON_COLUMNS] == "No internet service"
    inconsistent = (addon_says_none.ne(no_internet, axis=0)).any(axis=1)
    if inconsistent.any():
        problems.append(
            f"{int(inconsistent.sum())} rows where internet add-ons disagree with InternetService"
        )

    # Consistency: "No phone service" iff PhoneService == "No".
    phone_mismatch = (df["MultipleLines"] == "No phone service") != (df["PhoneService"] == "No")
    if phone_mismatch.any():
        problems.append(
            f"{int(phone_mismatch.sum())} rows where MultipleLines disagrees with PhoneService"
        )

    if problems:
        raise DataValidationError("Data validation failed:\n- " + "\n- ".join(problems))


# --- Public entry points -------------------------------------------------
def load_dataset(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load, clean, and validate the dataset. The single entry point for training code."""
    df = clean_data(load_raw_data(path))
    validate_data(df)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features X (identifier and target removed) and target y."""
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()


def split_train_test(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split, so both sets keep the overall churn rate.

    The test set is held out until final evaluation; all model selection
    happens with cross-validation on the training set.
    """
    return train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COLUMN], random_state=random_state
    )


def summarize(df: pd.DataFrame) -> str:
    """Short human-readable summary of a cleaned dataset."""
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    rates = df[TARGET_COLUMN].value_counts(normalize=True).sort_index()
    return (
        f"Rows: {len(df)}, columns: {df.shape[1]}\n"
        f"Features: {len(CATEGORICAL_COLUMNS)} categorical, {len(NUMERIC_COLUMNS)} numeric\n"
        f"Churn=0: {counts.get(0, 0)} ({rates.get(0, 0):.1%}), "
        f"Churn=1: {counts.get(1, 0)} ({rates.get(1, 0):.1%})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_raw_data()
    dataset = load_dataset()
    print("Validation passed.")
    print(summarize(dataset))
