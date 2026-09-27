"""Tests for data loading, cleaning, and validation.

Most tests use tiny hand-built DataFrames so they run without the real
dataset. One integration test uses the real file and is skipped if absent.
"""

import pandas as pd
import pytest

from churn import data
from churn.config import RAW_DATA_FILE


def make_raw_row(**overrides) -> dict:
    """A single valid raw row (as it appears in the CSV), with optional overrides."""
    row = {
        "customerID": "0001-AAAAA",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": "600.0",
        "Churn": "No",
    }
    row.update(overrides)
    return row


def make_raw_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows) or [make_raw_row()])


# --- clean_data ----------------------------------------------------------
def test_clean_converts_total_charges_and_target():
    df = data.clean_data(make_raw_df(make_raw_row(Churn="Yes")))
    assert df["TotalCharges"].iloc[0] == 600.0
    assert pd.api.types.is_float_dtype(df["TotalCharges"])
    assert df["Churn"].iloc[0] == 1


def test_clean_sets_blank_total_charges_to_zero_for_new_customers():
    raw = make_raw_df(make_raw_row(tenure=0, TotalCharges=" "))
    assert data.clean_data(raw)["TotalCharges"].iloc[0] == 0.0


def test_clean_rejects_blank_total_charges_for_existing_customers():
    raw = make_raw_df(make_raw_row(tenure=5, TotalCharges=" "))
    with pytest.raises(data.DataValidationError, match="TotalCharges"):
        data.clean_data(raw)


def test_clean_rejects_unknown_target_value():
    with pytest.raises(data.DataValidationError, match="Churn"):
        data.clean_data(make_raw_df(make_raw_row(Churn="Maybe")))


def test_clean_does_not_modify_input():
    raw = make_raw_df()
    data.clean_data(raw)
    assert raw["TotalCharges"].iloc[0] == "600.0"


# --- validate_data -------------------------------------------------------
def test_validate_accepts_valid_data():
    data.validate_data(data.clean_data(make_raw_df()))


def test_validate_reports_missing_columns():
    df = data.clean_data(make_raw_df()).drop(columns=["Contract"])
    with pytest.raises(data.DataValidationError, match="Missing required columns"):
        data.validate_data(df)


def test_validate_rejects_unknown_category():
    df = data.clean_data(make_raw_df(make_raw_row(Contract="Weekly")))
    with pytest.raises(data.DataValidationError, match="Contract"):
        data.validate_data(df)


def test_validate_rejects_duplicate_ids():
    df = data.clean_data(make_raw_df(make_raw_row(), make_raw_row()))
    with pytest.raises(data.DataValidationError, match="duplicate IDs"):
        data.validate_data(df)


def test_validate_rejects_negative_tenure():
    df = data.clean_data(make_raw_df(make_raw_row(tenure=-1)))
    with pytest.raises(data.DataValidationError, match="tenure"):
        data.validate_data(df)


def test_validate_rejects_inconsistent_internet_addons():
    row = make_raw_row(InternetService="No")  # add-ons still say "No"/"Yes"
    df = data.clean_data(make_raw_df(row))
    with pytest.raises(data.DataValidationError, match="InternetService"):
        data.validate_data(df)


def test_validate_reports_all_problems_at_once():
    row = make_raw_row(Contract="Weekly", gender="Unknown", tenure=-1)
    df = data.clean_data(make_raw_df(row))
    with pytest.raises(data.DataValidationError) as exc_info:
        data.validate_data(df)
    message = str(exc_info.value)
    assert "Contract" in message and "gender" in message and "tenure" in message


# --- split_features_target ----------------------------------------------
def test_split_excludes_identifier_and_target():
    X, y = data.split_features_target(data.clean_data(make_raw_df()))
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert list(X.columns) == data.FEATURE_COLUMNS
    assert len(X) == len(y)


# --- Real dataset (integration) -----------------------------------------
@pytest.mark.skipif(not RAW_DATA_FILE.exists(), reason="raw dataset not downloaded")
def test_real_dataset_loads_and_validates():
    df = data.load_dataset()
    assert df.shape == (7043, 21)
    assert df["TotalCharges"].notna().all()
    assert set(df["Churn"].unique()) == {0, 1}
