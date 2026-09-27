"""Tests for threshold analysis, threshold selection, and bootstrap intervals."""

import numpy as np
import pandas as pd
import pytest

from churn.evaluation import (
    bootstrap_metric_intervals,
    select_threshold_min_recall,
    threshold_table,
)

Y_TRUE = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1])
Y_PROBA = np.array([0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.6, 0.7, 0.8, 0.9])


def test_threshold_table_counts_and_rates():
    table = threshold_table(Y_TRUE, Y_PROBA, thresholds=[0.5])
    row = table.iloc[0]
    # Flagged: 0.6, 0.7, 0.8, 0.9 -> TP 3, FP 1; missed churner at 0.35 -> FN 1
    assert (row.tp, row.fp, row.fn, row.tn) == (3, 1, 1, 5)
    assert row.precision == pytest.approx(0.75)
    assert row.recall == pytest.approx(0.75)
    assert row.f1 == pytest.approx(0.75)
    assert row.predicted_positive_rate == pytest.approx(0.4)


def test_recall_does_not_increase_with_threshold():
    table = threshold_table(Y_TRUE, Y_PROBA)
    assert table["recall"].is_monotonic_decreasing
    assert table["predicted_positive_rate"].is_monotonic_decreasing


def test_no_flags_gives_zero_precision_without_error():
    row = threshold_table(Y_TRUE, Y_PROBA, thresholds=[0.95]).iloc[0]
    assert row.tp + row.fp == 0
    assert row.precision == 0.0 and row.f1 == 0.0


def test_select_threshold_meets_recall_target_with_best_precision():
    table = threshold_table(Y_TRUE, Y_PROBA, thresholds=[0.3, 0.34, 0.5, 0.65])
    # recall: 0.3 -> 1.0, 0.34 -> 1.0, 0.5 -> 0.75, 0.65 -> 0.5
    # precision: 0.3 -> 4/7, 0.34 -> 4/6 (best among recall 1.0)
    assert select_threshold_min_recall(table, min_recall=1.0) == 0.34
    assert select_threshold_min_recall(table, min_recall=0.75) == 0.5


def test_select_threshold_prefers_higher_threshold_on_precision_tie():
    table = pd.DataFrame({"threshold": [0.2, 0.3], "precision": [0.5, 0.5], "recall": [0.9, 0.9]})
    assert select_threshold_min_recall(table, min_recall=0.8) == 0.3


def test_select_threshold_raises_when_target_unreachable():
    table = threshold_table(Y_TRUE, Y_PROBA, thresholds=[0.95])
    with pytest.raises(ValueError, match="recall"):
        select_threshold_min_recall(table, min_recall=0.5)


def test_bootstrap_intervals_contain_estimate_and_are_reproducible():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    proba = np.clip(y * 0.4 + rng.uniform(0, 0.6, 300), 0, 1)

    first = bootstrap_metric_intervals(y, proba, threshold=0.5, n_resamples=200)
    second = bootstrap_metric_intervals(y, proba, threshold=0.5, n_resamples=200)

    pd.testing.assert_frame_equal(first, second)
    assert (first["lower"] <= first["estimate"]).all()
    assert (first["estimate"] <= first["upper"]).all()
