"""Model feature definitions and the sklearn preprocessing pipeline.

The preprocessor is always wrapped in a Pipeline together with the model, so
it is fitted only on training data (per CV fold during validation) and saved
as one artifact that the API later reuses. No preprocessing logic lives
outside this pipeline.
"""

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import CATEGORICAL_VALUES, INTERNET_ADDON_COLUMNS

# --- Feature groups -------------------------------------------------------
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Two-level features: encoded as a single 0/1 column.
BINARY_FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents", "PaperlessBilling"]

# Three or more levels: one column per level.
MULTICLASS_FEATURES = [
    "MultipleLines",
    "InternetService",
    *INTERNET_ADDON_COLUMNS,
    "Contract",
    "PaymentMethod",
]

# Excluded from the model, with the reason. customerID and the target are
# never part of the feature set (see churn.data.split_features_target).
EXCLUDED_FEATURES = {
    "PhoneService": "Fully determined by MultipleLines ('No phone service' level); Cramér's V = 1.0 in EDA.",
}


def model_features(include_total_charges: bool = True) -> list[str]:
    """Input columns the pipeline consumes, in order."""
    numeric = [c for c in NUMERIC_FEATURES if include_total_charges or c != "TotalCharges"]
    return numeric + BINARY_FEATURES + MULTICLASS_FEATURES


def build_preprocessor(
    include_total_charges: bool = True, scale_numeric: bool = True
) -> ColumnTransformer:
    """ColumnTransformer: scale numeric features, one-hot encode categoricals.

    Categories come from the validated schema rather than being learned from
    the training data, so the output columns are fixed regardless of which
    levels a given fold contains. Unknown categories raise an error instead of
    being silently encoded as all zeros; inputs are validated upstream.

    No imputer: the cleaned data is validated to contain no missing values, and
    an imputer here would hide upstream data problems.
    """
    numeric = [c for c in NUMERIC_FEATURES if include_total_charges or c != "TotalCharges"]
    categorical = BINARY_FEATURES + MULTICLASS_FEATURES

    encoder = OneHotEncoder(
        categories=[sorted(CATEGORICAL_VALUES[col]) for col in categorical],
        drop="if_binary",  # one column for two-level features; all levels otherwise
        handle_unknown="error",
        sparse_output=False,
    )
    numeric_step = StandardScaler() if scale_numeric else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_step, numeric),
            ("categorical", encoder, categorical),
        ],
        remainder="drop",  # customerID, PhoneService, and anything unexpected
        verbose_feature_names_out=False,
    )


def build_pipeline(model: BaseEstimator, **preprocessor_options) -> Pipeline:
    """Preprocessing + model as a single estimator (fit, predict, and save together)."""
    return Pipeline([
        ("preprocess", build_preprocessor(**preprocessor_options)),
        ("model", model),
    ])
