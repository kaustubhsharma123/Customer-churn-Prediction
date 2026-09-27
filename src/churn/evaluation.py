"""Classification metrics for churn models.

Threshold-free metrics (ROC-AUC, PR-AUC, Brier, log loss) evaluate the
predicted probabilities; precision/recall/F1 depend on the decision threshold
and are reported at an explicit threshold, never implicitly.
"""

import numpy as np
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
