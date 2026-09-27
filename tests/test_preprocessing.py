"""Tests for feature groups, the preprocessing pipeline, and the train/test split."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from churn import data, preprocessing

# 3 numeric + 5 binary (1 column each) + multi-level one-hot columns:
# MultipleLines 3, InternetService 3, 6 add-ons x 3, Contract 3, PaymentMethod 4
EXPECTED_N_OUTPUT_COLUMNS = 3 + 5 + (3 + 3 + 18 + 3 + 4)


# --- Feature groups ---------------------------------------------------------
def test_feature_groups_cover_every_feature_exactly_once():
    grouped = (
        preprocessing.NUMERIC_FEATURES
        + preprocessing.BINARY_FEATURES
        + preprocessing.MULTICLASS_FEATURES
        + list(preprocessing.EXCLUDED_FEATURES)
    )
    assert sorted(grouped) == sorted(data.FEATURE_COLUMNS)
    assert len(grouped) == len(set(grouped))


def test_binary_and_multiclass_groups_match_schema_levels():
    for col in preprocessing.BINARY_FEATURES:
        assert len(data.CATEGORICAL_VALUES[col]) == 2
    for col in preprocessing.MULTICLASS_FEATURES:
        assert len(data.CATEGORICAL_VALUES[col]) > 2


def test_model_features_exclude_identifier_target_and_phone_service():
    features = preprocessing.model_features()
    for col in ("customerID", "Churn", "PhoneService"):
        assert col not in features
    assert "TotalCharges" not in preprocessing.model_features(include_total_charges=False)


# --- Preprocessor -----------------------------------------------------------
def test_preprocessor_output_shape_and_names(clean_df):
    X, _ = data.split_features_target(clean_df)
    pre = preprocessing.build_preprocessor().fit(X)
    names = list(pre.get_feature_names_out())

    assert pre.transform(X).shape == (len(X), EXPECTED_N_OUTPUT_COLUMNS)
    assert "Partner_Yes" in names and "Partner_No" not in names  # binary -> one column
    assert {"Contract_Month-to-month", "Contract_One year", "Contract_Two year"} <= set(names)
    assert not any(n.startswith(("customerID", "PhoneService")) for n in names)


def test_preprocessor_without_total_charges(clean_df):
    X, _ = data.split_features_target(clean_df)
    pre = preprocessing.build_preprocessor(include_total_charges=False).fit(X)
    assert "TotalCharges" not in pre.get_feature_names_out()
    assert pre.transform(X).shape[1] == EXPECTED_N_OUTPUT_COLUMNS - 1


def test_output_columns_do_not_depend_on_levels_seen_in_training(clean_df):
    X, _ = data.split_features_target(clean_df)
    fitted_on_one_row = preprocessing.build_preprocessor().fit(X.iloc[[0]])
    assert fitted_on_one_row.transform(X).shape[1] == EXPECTED_N_OUTPUT_COLUMNS


def test_output_is_finite(clean_df):
    X, _ = data.split_features_target(clean_df)
    assert np.isfinite(preprocessing.build_preprocessor().fit_transform(X)).all()


def test_unknown_category_raises(clean_df):
    X, _ = data.split_features_target(clean_df)
    pre = preprocessing.build_preprocessor().fit(X)
    bad = X.iloc[[0]].copy()
    bad["Contract"] = "Weekly"
    with pytest.raises(ValueError):
        pre.transform(bad)


def test_missing_input_column_raises(clean_df):
    X, _ = data.split_features_target(clean_df)
    pre = preprocessing.build_preprocessor().fit(X)
    with pytest.raises(ValueError):
        pre.transform(X.drop(columns=["tenure"]))


def test_scaler_uses_training_statistics_only(clean_df):
    X, _ = data.split_features_target(clean_df)
    X_train, X_other = X.iloc[:100], X.iloc[100:]
    pre = preprocessing.build_preprocessor().fit(X_train)

    scaler = pre.named_transformers_["numeric"]
    assert scaler.mean_ == pytest.approx(X_train[preprocessing.NUMERIC_FEATURES].mean().to_numpy())
    # Transforming other data must not refit: its scaled mean is not forced to zero.
    tenure_scaled = pre.transform(X_other)[:, 0]
    expected = (X_other["tenure"] - X_train["tenure"].mean()) / X_train["tenure"].std(ddof=0)
    assert tenure_scaled == pytest.approx(expected.to_numpy())


def test_unscaled_option_passes_numeric_values_through(clean_df):
    X, _ = data.split_features_target(clean_df)
    out = preprocessing.build_preprocessor(scale_numeric=False).fit_transform(X)
    assert out[:, 0] == pytest.approx(X["tenure"].to_numpy())


# --- Full pipeline ----------------------------------------------------------
def test_pipeline_fits_and_returns_probabilities(clean_df):
    X, y = data.split_features_target(clean_df)
    pipe = preprocessing.build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


# --- Train/test split -------------------------------------------------------
def test_split_is_stratified_disjoint_and_reproducible(clean_df):
    train, test = data.split_train_test(clean_df, test_size=0.25)
    train_again, test_again = data.split_train_test(clean_df, test_size=0.25)

    assert len(train) + len(test) == len(clean_df)
    assert set(train["customerID"]).isdisjoint(test["customerID"])
    assert abs(train["Churn"].mean() - test["Churn"].mean()) < 0.03
    assert list(test["customerID"]) == list(test_again["customerID"])
