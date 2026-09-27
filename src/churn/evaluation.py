"""Classification metrics for churn models.

Threshold-free metrics (ROC-AUC, PR-AUC, Brier, log loss) evaluate the
predicted probabilities; precision/recall/F1 depend on the decision threshold
and are reported at an explicit threshold, never implicitly.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLD = 0.5


def classification_metrics(
    y_true, y_proba, threshold: float = DEFAULT_THRESHOLD
) -> dict[str, float]:
    """Probability-based and threshold-based metrics for the churn (positive) class."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
    }


def confusion_at_threshold(y_true, y_proba, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """2x2 confusion matrix [[TN, FP], [FN, TP]] at the given threshold."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


# --- Threshold analysis ----------------------------------------------------
DEFAULT_THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def threshold_table(y_true, y_proba, thresholds=DEFAULT_THRESHOLD_GRID) -> pd.DataFrame:
    """Precision, recall, F1, and predicted positive rate at each threshold."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    rows = []
    for threshold in thresholds:
        (tn, fp), (fn, tp) = confusion_at_threshold(y_true, y_proba, threshold)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_positive_rate": (tp + fp) / len(y_true),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })
    return pd.DataFrame(rows)


def select_threshold_min_recall(table: pd.DataFrame, min_recall: float) -> float:
    """Highest-precision threshold whose recall is at least `min_recall`.

    Ties in precision go to the higher threshold (fewer customers flagged).
    Raises if no threshold reaches the recall target.
    """
    eligible = table[table["recall"] >= min_recall]
    if eligible.empty:
        raise ValueError(f"No threshold reaches recall >= {min_recall}")
    best = eligible.sort_values(["precision", "threshold"], ascending=False).iloc[0]
    return float(best["threshold"])


def bootstrap_metric_intervals(
    y_true,
    y_proba,
    threshold: float,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> pd.DataFrame:
    """Percentile bootstrap intervals for each metric (resampling evaluation rows).

    Shows how much a single test-set estimate could vary from sampling alone.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    rng = np.random.default_rng(random_state)
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():  # need both classes for AUC
            continue
        samples.append(classification_metrics(y_true[idx], y_proba[idx], threshold))
    samples = pd.DataFrame(samples)
    alpha = (1 - confidence) / 2
    return pd.DataFrame({
        "estimate": classification_metrics(y_true, y_proba, threshold),
        "lower": samples.quantile(alpha),
        "upper": samples.quantile(1 - alpha),
    })
