"""Explainability for the fitted churn pipeline.

- SHAP (TreeExplainer, tree-path-dependent) on the pipeline's own fitted
  preprocessor output: exact, additive contributions in log-odds space.
- One-hot columns are mapped back to their original feature and summed
  (valid because SHAP values are additive).
- Grouped permutation importance on held-out CV folds, to measure the joint
  importance of correlated features.

Contributions describe how the *model* uses features (association), not
what causes churn.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from churn.config import RANDOM_STATE
from churn.data import INTERNET_ADDON_COLUMNS

# Features that carry overlapping information (see EDA): permuted together.
CORRELATED_GROUPS: dict[str, list[str]] = {
    "tenure + charges": ["tenure", "MonthlyCharges", "TotalCharges"],
    "internet service + add-ons": ["InternetService", *INTERNET_ADDON_COLUMNS],
}


# --- Mapping encoded columns back to original features --------------------
def encoded_to_original(preprocessor: ColumnTransformer) -> dict[str, str]:
    """Map each encoded output column of a *fitted* preprocessor to its input feature."""
    mapping: dict[str, str] = {}
    for _, transformer, columns in preprocessor.transformers_:
        if transformer == "drop" or not len(columns):
            continue
        if isinstance(transformer, OneHotEncoder):
            names = iter(transformer.get_feature_names_out(columns))
            for col, categories, dropped in zip(columns, transformer.categories_, transformer.drop_idx_):
                n_out = len(categories) - (dropped is not None)
                for _ in range(n_out):
                    mapping[next(names)] = col
        else:  # scaler / passthrough: one output per input column
            for col in columns:
                mapping[col] = col
    output_names = list(preprocessor.get_feature_names_out())
    if list(mapping) != output_names:
        raise ValueError("Encoded-column mapping does not match the preprocessor output")
    return mapping


def encode(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Model-ready features, produced by the pipeline's own fitted preprocessor."""
    preprocessor = pipeline.named_steps["preprocess"]
    return pd.DataFrame(
        preprocessor.transform(X), columns=preprocessor.get_feature_names_out(), index=X.index
    )


# --- SHAP ---------------------------------------------------------------------
def make_explainer(pipeline: Pipeline) -> shap.TreeExplainer:
    """Exact tree SHAP explainer for the pipeline's (fitted) tree model."""
    return shap.TreeExplainer(pipeline.named_steps["model"])


@dataclass
class ShapResult:
    values: pd.DataFrame  # rows x original features, log-odds contributions
    base_value: float  # model's expected log-odds (the starting point of every explanation)


def grouped_shap_values(pipeline: Pipeline, explainer: shap.TreeExplainer, X: pd.DataFrame) -> ShapResult:
    """SHAP values per original feature (one-hot columns summed), in log-odds."""
    encoded = encode(pipeline, X)
    values = np.asarray(explainer.shap_values(encoded.to_numpy()))
    per_column = pd.DataFrame(values, columns=encoded.columns, index=X.index)
    mapping = encoded_to_original(pipeline.named_steps["preprocess"])
    grouped = per_column.T.groupby(mapping, sort=False).sum().T
    base_value = float(np.ravel(explainer.expected_value)[0])
    return ShapResult(values=grouped, base_value=base_value)


def global_shap_importance(values: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP value (log-odds) per feature, largest first."""
    return values.abs().mean().sort_values(ascending=False)


@dataclass
class LocalExplanation:
    base_value: float  # log-odds before any feature is considered
    predicted_log_odds: float
    predicted_probability: float
    contributions: pd.DataFrame  # feature, value, contribution (log-odds), largest |contribution| first

    @property
    def base_probability(self) -> float:
        """expit(base log-odds). Not the average predicted probability (log-odds are averaged)."""
        return float(expit(self.base_value))

    def top(self, k: int) -> pd.DataFrame:
        """Top-k contributions plus one row summing all remaining features."""
        head = self.contributions.head(k)
        rest = self.contributions.iloc[k:]
        if rest.empty:
            return head
        other = pd.DataFrame([{
            "feature": f"{len(rest)} other features", "value": "", "contribution": rest["contribution"].sum(),
        }])
        return pd.concat([head, other], ignore_index=True)


def explain_prediction(pipeline: Pipeline, explainer: shap.TreeExplainer, row: pd.DataFrame) -> LocalExplanation:
    """Explain one customer's prediction (row: single-row DataFrame of raw features)."""
    if len(row) != 1:
        raise ValueError("explain_prediction expects exactly one row")
    shap_result = grouped_shap_values(pipeline, explainer, row)
    contributions = shap_result.values.iloc[0]
    table = pd.DataFrame({
        "feature": contributions.index,
        "value": [row.iloc[0][f] for f in contributions.index],
        "contribution": contributions.to_numpy(),
    })
    table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    log_odds = shap_result.base_value + contributions.sum()
    return LocalExplanation(
        base_value=shap_result.base_value,
        predicted_log_odds=float(log_odds),
        predicted_probability=float(pipeline.predict_proba(row)[0, 1]),
        contributions=table,
    )


# --- Grouped permutation importance -------------------------------------------
def cv_permutation_importance(
    make_pipeline: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    groups: dict[str, list[str]],
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Drop in held-out ROC-AUC when each feature group is shuffled (jointly).

    For each CV fold, a fresh pipeline is fitted on the training part; each
    group's columns are shuffled together (same row permutation) in the
    validation part. Joint shuffling keeps within-group relationships while
    breaking the group's link to the target.
    """
    rng = np.random.default_rng(random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        pipeline = make_pipeline().fit(X.iloc[train_idx], y.iloc[train_idx])
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        baseline = roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])
        for name, columns in groups.items():
            for _ in range(n_repeats):
                shuffled = X_val.copy()
                order = rng.permutation(len(X_val))
                shuffled[columns] = X_val[columns].to_numpy()[order]
                score = roc_auc_score(y_val, pipeline.predict_proba(shuffled)[:, 1])
                rows.append({"group": name, "fold": fold, "auc_drop": baseline - score})
    result = pd.DataFrame(rows).groupby("group")["auc_drop"].agg(["mean", "std"])
    return result.sort_values("mean", ascending=False)
