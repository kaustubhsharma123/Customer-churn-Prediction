"""Shared fixtures."""

import numpy as np
import pandas as pd
import pytest

from churn import data


def make_clean_dataset(n_rows: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic *cleaned* dataset that satisfies every schema invariant."""
    rng = np.random.default_rng(seed)
    pick = lambda values, size=n_rows: rng.choice(sorted(values), size=size)  # noqa: E731

    internet = pick(data.CATEGORICAL_VALUES["InternetService"])
    phone = pick(data.CATEGORICAL_VALUES["PhoneService"])
    tenure = rng.integers(0, 73, n_rows)
    monthly = rng.uniform(18.25, 118.75, n_rows).round(2)

    df = pd.DataFrame({
        "customerID": [f"{i:04d}-ABCDE" for i in range(n_rows)],
        "gender": pick(data.CATEGORICAL_VALUES["gender"]),
        "SeniorCitizen": pick(data.CATEGORICAL_VALUES["SeniorCitizen"]),
        "Partner": pick({"Yes", "No"}),
        "Dependents": pick({"Yes", "No"}),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", pick({"Yes", "No"})),
        "InternetService": internet,
        **{
            col: np.where(internet == "No", "No internet service", pick({"Yes", "No"}))
            for col in data.INTERNET_ADDON_COLUMNS
        },
        "Contract": pick(data.CATEGORICAL_VALUES["Contract"]),
        "PaperlessBilling": pick({"Yes", "No"}),
        "PaymentMethod": pick(data.CATEGORICAL_VALUES["PaymentMethod"]),
        "MonthlyCharges": monthly,
        "TotalCharges": (tenure * monthly).round(2).astype(float),
        "Churn": rng.choice([0, 1], size=n_rows, p=[0.73, 0.27]),
    })
    data.validate_data(df)
    return df[data.EXPECTED_COLUMNS]


@pytest.fixture
def clean_df() -> pd.DataFrame:
    return make_clean_dataset()
