"""Baseline model definitions and cross-validated comparison.

Every model is wrapped in the shared preprocessing Pipeline, so the
preprocessor is re-fitted inside each training fold (no leakage into the
validation fold). Only the training set is passed in here; the test set is
reserved for final evaluation.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from churn.config import RANDOM_STATE
from churn.evaluation import DEFAULT_THRESHOLD, classification_metrics
from churn.preprocessing import build_pipeline


@dataclass(frozen=True)
class ModelSpec:
    build: Callable[[], BaseEstimator]
    scale_numeric: bool  # linear models need scaling; trees do not


# Untuned baselines with library defaults (plus a fixed seed). Tuning is Phase 7.
BASELINE_MODELS: dict[str, ModelSpec] = {
    "LogisticRegression": ModelSpec(
        build=lambda: LogisticRegression(max_iter=2000),
        scale_numeric=True,
    ),
    "RandomForest": ModelSpec(
        build=lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
        scale_numeric=False,
    ),
    "HistGradientBoosting": ModelSpec(
        build=lambda: HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        scale_numeric=False,
    ),
}


def make_baseline_pipeline(name: str) -> Pipeline:
    """Fresh, unfitted preprocessing + model pipeline for a named baseline."""
    if name not in BASELINE_MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {list(BASELINE_MODELS)}")
    spec = BASELINE_MODELS[name]
    return build_pipeline(spec.build(), scale_numeric=spec.scale_numeric)


@dataclass
class CVResult:
    fold_metrics: pd.DataFrame  # one row per (model, fold)
    oof_predictions: pd.DataFrame  # one row per (model, repeat, training row), with its fold id


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    model_names: list[str] | None = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    threshold: float = DEFAULT_THRESHOLD,
    random_state: int = RANDOM_STATE,
) -> CVResult:
    """Repeated stratified k-fold CV; all models see identical folds.

    Records validation metrics per fold, the training-fold ROC-AUC (to spot
    overfitting), and every out-of-fold predicted probability.
    """
    model_names = model_names or list(BASELINE_MODELS)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    folds = list(cv.split(X, y))

    metric_rows, oof_frames = [], []
    for name in model_names:
        for fold_id, (train_idx, val_idx) in enumerate(folds):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            pipeline = make_baseline_pipeline(name).fit(X_tr, y_tr)
            val_proba = pipeline.predict_proba(X_val)[:, 1]
            train_proba = pipeline.predict_proba(X_tr)[:, 1]

            metric_rows.append({
                "model": name,
                "repeat": fold_id // n_splits,
                "fold": fold_id,
                **classification_metrics(y_val, val_proba, threshold),
                "train_roc_auc": roc_auc_score(y_tr, train_proba),
            })
            oof_frames.append(pd.DataFrame({
                "model": name,
                "repeat": fold_id // n_splits,
                "fold": fold_id,
                "row_index": X_val.index,
                "y_true": y_val.to_numpy(),
                "proba": val_proba,
            }))

    return CVResult(
        fold_metrics=pd.DataFrame(metric_rows),
        oof_predictions=pd.concat(oof_frames, ignore_index=True),
    )


def summarize_cv(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric per model."""
    metrics = fold_metrics.drop(columns=["repeat", "fold"])
    return metrics.groupby("model").agg(["mean", "std"])
