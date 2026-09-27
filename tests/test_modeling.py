"""Tests for baseline model pipelines and cross-validation."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn import data, modeling


@pytest.fixture
def train_xy(clean_df):
    return data.split_features_target(clean_df)


@pytest.mark.parametrize("name", list(modeling.BASELINE_MODELS))
def test_baseline_pipeline_structure(name):
    pipeline = modeling.make_baseline_pipeline(name)
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocess", "model"]


def test_pipelines_are_independent_instances():
    first = modeling.make_baseline_pipeline("LogisticRegression")
    second = modeling.make_baseline_pipeline("LogisticRegression")
    assert first is not second
    assert first.named_steps["model"] is not second.named_steps["model"]


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        modeling.make_baseline_pipeline("SVM")


def test_cross_validation_outputs(train_xy):
    X, y = train_xy
    result = modeling.cross_validate_models(X, y, n_splits=3, n_repeats=2)

    n_models = len(modeling.BASELINE_MODELS)
    assert len(result.fold_metrics) == n_models * 3 * 2
    assert {"roc_auc", "pr_auc", "precision", "recall", "f1", "train_roc_auc"} <= set(result.fold_metrics)

    oof = result.oof_predictions
    assert oof["proba"].between(0, 1).all()
    # Every training row is predicted exactly once per model and repeat (true out-of-fold).
    counts = oof.groupby(["model", "repeat"])["row_index"].agg(["size", "nunique"])
    assert (counts["size"] == len(X)).all()
    assert (counts["nunique"] == len(X)).all()


def test_cross_validation_is_reproducible(train_xy):
    X, y = train_xy
    first = modeling.cross_validate_models(X, y, model_names=["HistGradientBoosting"], n_splits=3, n_repeats=1)
    second = modeling.cross_validate_models(X, y, model_names=["HistGradientBoosting"], n_splits=3, n_repeats=1)
    pd.testing.assert_frame_equal(first.fold_metrics, second.fold_metrics)


def test_summarize_cv(train_xy):
    X, y = train_xy
    result = modeling.cross_validate_models(X, y, model_names=["LogisticRegression"], n_splits=3, n_repeats=1)
    summary = modeling.summarize_cv(result.fold_metrics)
    assert summary.loc["LogisticRegression", ("roc_auc", "mean")] == pytest.approx(
        result.fold_metrics["roc_auc"].mean()
    )
