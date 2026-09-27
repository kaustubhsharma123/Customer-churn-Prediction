"""Tests for hyperparameter search and tuned-pipeline evaluation."""

from functools import partial

import pytest
from sklearn.model_selection import ParameterGrid

from churn import data, modeling, tuning


@pytest.fixture
def train_xy(clean_df):
    return data.split_features_target(clean_df)


def test_search_spaces_use_valid_pipeline_parameters():
    for name, space in tuning.SEARCH_SPACES.items():
        valid = modeling.make_baseline_pipeline(name).get_params()
        assert set(space.params) <= set(valid), name


def test_search_spaces_stay_small():
    for space in tuning.SEARCH_SPACES.values():
        size = space.n_iter if space.n_iter is not None else len(ParameterGrid(space.params))
        assert size <= 30


def test_tuning_folds_differ_from_evaluation_folds():
    assert tuning.TUNING_CV_SEED != modeling.RANDOM_STATE


def test_make_tuned_pipeline_applies_params_without_mutating_baseline():
    tuned = tuning.make_tuned_pipeline("LogisticRegression", {"model__C": 0.1})
    assert tuned.named_steps["model"].C == 0.1
    assert modeling.make_baseline_pipeline("LogisticRegression").named_steps["model"].C == 1.0


def test_grid_search_ranks_by_roc_auc(train_xy):
    X, y = train_xy
    space = tuning.SearchSpace({"model__C": [0.01, 1.0], "model__solver": ["liblinear"]})
    result = tuning.tune_model("LogisticRegression", X, y, n_splits=3, n_jobs=1, search_space=space)

    assert len(result.results) == 2
    assert result.results["cv_roc_auc"].is_monotonic_decreasing
    assert result.best_params == result.results.loc[0, "params"]
    assert {"cv_pr_auc", "train_roc_auc", "overfit_gap"} <= set(result.results)


def test_random_search_is_reproducible(train_xy):
    X, y = train_xy
    space = tuning.SearchSpace(
        {"model__max_leaf_nodes": [7, 15, 31], "model__min_samples_leaf": [5, 10, 20]}, n_iter=3
    )
    first = tuning.tune_model("HistGradientBoosting", X, y, n_splits=3, n_jobs=1, search_space=space)
    second = tuning.tune_model("HistGradientBoosting", X, y, n_splits=3, n_jobs=1, search_space=space)
    assert first.results["params"].tolist() == second.results["params"].tolist()
    assert first.results["cv_roc_auc"].tolist() == second.results["cv_roc_auc"].tolist()


def test_cross_validate_accepts_custom_pipeline_factories(train_xy):
    X, y = train_xy
    factories = {
        "LR_C0.1": partial(tuning.make_tuned_pipeline, "LogisticRegression", {"model__C": 0.1}),
        "LR_baseline": partial(modeling.make_baseline_pipeline, "LogisticRegression"),
    }
    result = modeling.cross_validate_models(X, y, n_splits=3, n_repeats=1, pipeline_factories=factories)
    assert set(result.fold_metrics["model"]) == set(factories)
    assert len(result.oof_predictions) == 2 * len(X)
