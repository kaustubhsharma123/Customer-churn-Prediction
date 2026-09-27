"""Tests for explainability utilities."""

from functools import partial

import numpy as np
import pandas as pd
import pytest

from churn import data, explain, preprocessing, tuning

PARAMS = {"model__max_iter": 30, "model__max_leaf_nodes": 7}


@pytest.fixture
def fitted(clean_df):
    X, y = data.split_features_target(clean_df)
    pipeline = tuning.make_tuned_pipeline("HistGradientBoosting", PARAMS).fit(X, y)
    return pipeline, explain.make_explainer(pipeline), X, y


def test_encoded_mapping_covers_every_output_column(fitted):
    pipeline, _, _, _ = fitted
    mapping = explain.encoded_to_original(pipeline.named_steps["preprocess"])
    assert list(mapping) == list(pipeline.named_steps["preprocess"].get_feature_names_out())
    assert set(mapping.values()) == set(preprocessing.model_features())
    assert mapping["Contract_Two year"] == "Contract"
    assert mapping["PaymentMethod_Bank transfer (automatic)"] == "PaymentMethod"
    assert mapping["Partner_Yes"] == "Partner"
    assert mapping["tenure"] == "tenure"


def test_encode_uses_pipeline_preprocessor(fitted):
    pipeline, _, X, _ = fitted
    encoded = explain.encode(pipeline, X)
    expected = pipeline.named_steps["preprocess"].transform(X)
    np.testing.assert_array_equal(encoded.to_numpy(), expected)


def test_grouped_shap_is_additive_to_model_output(fitted):
    pipeline, explainer, X, _ = fitted
    result = explain.grouped_shap_values(pipeline, explainer, X.head(50))

    assert list(result.values.columns) == preprocessing.model_features()
    margin = pipeline.decision_function(X.head(50))
    np.testing.assert_allclose(result.values.sum(axis=1) + result.base_value, margin, atol=1e-6)


def test_global_importance_sorted_and_non_negative(fitted):
    pipeline, explainer, X, _ = fitted
    importance = explain.global_shap_importance(explain.grouped_shap_values(pipeline, explainer, X).values)
    assert (importance >= 0).all()
    assert importance.is_monotonic_decreasing


def test_local_explanation_matches_prediction(fitted):
    pipeline, explainer, X, _ = fitted
    row = X.iloc[[3]]
    local = explain.explain_prediction(pipeline, explainer, row)

    assert local.predicted_probability == pytest.approx(pipeline.predict_proba(row)[0, 1])
    assert local.base_value + local.contributions["contribution"].sum() == pytest.approx(local.predicted_log_odds)
    assert 1 / (1 + np.exp(-local.predicted_log_odds)) == pytest.approx(local.predicted_probability)
    assert local.contributions["contribution"].abs().is_monotonic_decreasing
    # Raw (unencoded) feature values are shown.
    contract = local.contributions.set_index("feature").loc["Contract", "value"]
    assert contract == row.iloc[0]["Contract"]


def test_local_top_k_preserves_total(fitted):
    pipeline, explainer, X, _ = fitted
    local = explain.explain_prediction(pipeline, explainer, X.iloc[[0]])
    top = local.top(5)
    assert len(top) == 6
    assert top["contribution"].sum() == pytest.approx(local.contributions["contribution"].sum())


def test_explain_prediction_rejects_multiple_rows(fitted):
    pipeline, explainer, X, _ = fitted
    with pytest.raises(ValueError, match="one row"):
        explain.explain_prediction(pipeline, explainer, X.head(2))


def test_permutation_importance_detects_signal_and_ignores_noise(clean_df):
    df = clean_df.copy()
    rng = np.random.default_rng(0)
    df["Churn"] = (df["Contract"] == "Month-to-month").astype(int)  # target depends on Contract only
    flip = rng.random(len(df)) < 0.05
    df.loc[flip, "Churn"] = 1 - df.loc[flip, "Churn"]
    X, y = data.split_features_target(df)

    make = partial(tuning.make_tuned_pipeline, "HistGradientBoosting", PARAMS)
    result = explain.cv_permutation_importance(
        make, X, y, groups={"Contract": ["Contract"], "gender": ["gender"]}, n_splits=3, n_repeats=3
    )
    assert result.index[0] == "Contract"
    assert result.loc["Contract", "mean"] > 0.2
    assert abs(result.loc["gender", "mean"]) < 0.05


def test_joint_permutation_keeps_rows_consistent_within_group(clean_df):
    X, y = data.split_features_target(clean_df)
    make = partial(tuning.make_tuned_pipeline, "HistGradientBoosting", PARAMS)
    group = explain.CORRELATED_GROUPS["internet service + add-ons"]
    # Must not raise and must return a finite importance for the whole group.
    result = explain.cv_permutation_importance(make, X, y, groups={"g": group}, n_splits=3, n_repeats=1)
    assert np.isfinite(result.loc["g", "mean"])
    assert isinstance(result, pd.DataFrame)
