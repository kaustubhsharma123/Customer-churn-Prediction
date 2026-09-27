"""Limited hyperparameter search for the baseline models (training data only).

Search uses its own stratified 5-fold split (seed RANDOM_STATE + 1) so that the
folds used to *choose* hyperparameters differ from the repeated folds later
used to *compare* tuned and baseline models. The preprocessing Pipeline is
part of every candidate, so it is re-fitted inside each fold.

class_weight is deliberately not searched: re-weighting classes distorts
predicted probabilities (hurting calibration), and the precision/recall
trade-off is handled by the decision threshold instead.
"""

from dataclasses import dataclass

import pandas as pd
from scipy.stats import loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from churn.config import RANDOM_STATE
from churn.modeling import make_baseline_pipeline

TUNING_CV_SEED = RANDOM_STATE + 1
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
PRIMARY_METRIC = "roc_auc"


@dataclass(frozen=True)
class SearchSpace:
    params: dict
    n_iter: int | None = None  # None -> exhaustive grid; otherwise random search


SEARCH_SPACES: dict[str, SearchSpace] = {
    # Regularization strength and type. liblinear supports both L1 and L2.
    "LogisticRegression": SearchSpace({
        "model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        "model__l1_ratio": [0.0, 1.0],  # 0 = L2, 1 = L1
        "model__solver": ["liblinear"],
    }),
    # Regularize the boosted trees: fewer/smaller leaves, larger leaves, L2, slower learning.
    "HistGradientBoosting": SearchSpace(
        {
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_iter": [100, 200, 300],
            "model__max_leaf_nodes": [7, 15, 31],
            "model__max_depth": [None, 3, 5],
            "model__min_samples_leaf": [20, 50, 100, 200],
            "model__l2_regularization": loguniform(1e-3, 10.0),
        },
        n_iter=25,
    ),
    # Phase 5 RF memorised the training folds (train ROC-AUC 1.0); constrain leaf size and features.
    "RandomForest": SearchSpace({
        "model__min_samples_leaf": [1, 5, 10, 20, 50],
        "model__max_features": ["sqrt", 0.3, 0.5],
        "model__n_jobs": [1],  # parallelism comes from the search itself
    }),
}


@dataclass
class TuningResult:
    model: str
    best_params: dict
    results: pd.DataFrame  # one row per candidate, best first


def tune_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = TUNING_CV_SEED,
    n_jobs: int = -1,
    search_space: SearchSpace | None = None,
) -> TuningResult:
    """Search one model's hyperparameters with stratified CV; rank by ROC-AUC."""
    space = search_space or SEARCH_SPACES[name]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    common = dict(scoring=SCORING, refit=False, cv=cv, n_jobs=n_jobs, return_train_score=True)

    pipeline = make_baseline_pipeline(name)
    if space.n_iter is None:
        search = GridSearchCV(pipeline, space.params, **common)
    else:
        search = RandomizedSearchCV(
            pipeline, space.params, n_iter=space.n_iter, random_state=random_state, **common
        )
    search.fit(X, y)

    raw = pd.DataFrame(search.cv_results_)
    results = pd.DataFrame({
        "params": raw["params"],
        "cv_roc_auc": raw["mean_test_roc_auc"],
        "cv_roc_auc_std": raw["std_test_roc_auc"],
        "cv_pr_auc": raw["mean_test_pr_auc"],
        "train_roc_auc": raw["mean_train_roc_auc"],
    })
    results["overfit_gap"] = results["train_roc_auc"] - results["cv_roc_auc"]
    results = results.sort_values(f"cv_{PRIMARY_METRIC}", ascending=False).reset_index(drop=True)
    return TuningResult(model=name, best_params=results.loc[0, "params"], results=results)


def make_tuned_pipeline(name: str, params: dict) -> Pipeline:
    """Fresh, unfitted pipeline for `name` with the given hyperparameters."""
    return make_baseline_pipeline(name).set_params(**params)
