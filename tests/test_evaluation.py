"""Tests for classification metrics."""

import numpy as np
import pytest

from churn.evaluation import classification_metrics, confusion_at_threshold

Y_TRUE = np.array([0, 0, 0, 1, 1])


def test_perfect_predictions():
    metrics = classification_metrics(Y_TRUE, [0.1, 0.2, 0.3, 0.8, 0.9])
    for name in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
        assert metrics[name] == pytest.approx(1.0)
    assert metrics["brier"] < 0.05


def test_threshold_changes_threshold_metrics_only():
    proba = [0.1, 0.4, 0.6, 0.7, 0.9]
    at_half = classification_metrics(Y_TRUE, proba, threshold=0.5)
    at_high = classification_metrics(Y_TRUE, proba, threshold=0.65)

    assert at_half["precision"] == pytest.approx(2 / 3)  # 0.6 is a false positive
    assert at_high["precision"] == pytest.approx(1.0)
    for name in ("roc_auc", "pr_auc", "brier", "log_loss"):
        assert at_half[name] == at_high[name]


def test_no_positive_predictions_does_not_crash():
    metrics = classification_metrics(Y_TRUE, [0.1] * 5)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_confusion_layout():
    matrix = confusion_at_threshold(Y_TRUE, [0.1, 0.6, 0.2, 0.3, 0.9])
    # [[TN, FP], [FN, TP]]
    assert matrix.tolist() == [[2, 1], [1, 1]]
