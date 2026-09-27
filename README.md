# Customer Churn Prediction System

> **Status: work in progress.** Built incrementally, one phase at a time. No model has been trained yet; results will be added once they exist.

An end-to-end machine learning system that estimates the probability a customer will churn, explains the main factors behind each prediction, and serves predictions through a FastAPI backend and a Streamlit frontend.

## Planned architecture

```
raw data → validation/cleaning → sklearn Pipeline (preprocessing + model) → saved .joblib
                                                                                 │
                                                  Streamlit UI → FastAPI → load pipeline → prediction
```

Training and inference share the **same saved pipeline**, so preprocessing is never duplicated between them.

## Project structure

```
├── data/
│   ├── raw/            # original dataset (never modified)
│   └── processed/      # reproducible derived data
├── notebooks/          # exploratory analysis
├── src/churn/          # reusable package: config, data, preprocessing, training, prediction
├── api/                # FastAPI backend
├── app/                # Streamlit frontend
├── models/             # saved pipeline artifacts
├── reports/figures/    # generated plots
├── tests/              # pytest tests
├── pyproject.toml      # makes src/churn installable
└── requirements.txt    # dependencies
```

## Setup (Windows, Python 3.12)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux: activate with `source venv/bin/activate`.

Verify the installation:

```powershell
python -c "import churn, sklearn, pandas, fastapi, streamlit, shap; print('OK')"
pytest
```

## Dataset

**IBM Telco Customer Churn** (sample data): 7,043 customers of a fictional telecom company, 19 features plus `customerID` and the target `Churn` (26.5% churned).

- **Source:** [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d) (`data/Telco-Customer-Churn.csv`). The repository is Apache-2.0 licensed; the license does not explicitly address the data file, and IBM has not published separate data terms that we could find. Also mirrored on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- **Not committed to this repo.** Download and validate it (SHA-256 verified):

```powershell
python -m churn.data
```

Known data-quality issues handled in `src/churn/data.py`:
- `TotalCharges` is stored as text; 11 customers with `tenure == 0` have a blank value, set to `0.0` (not yet billed).
- `customerID` is an identifier and is excluded from model features.

## Tech stack

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · SHAP · FastAPI · Pydantic · Streamlit · Joblib · Pytest

## Roadmap

- [x] Phase 1 — Project foundation
- [ ] Phase 2 — Data
- [ ] Phase 3 — Exploratory data analysis
- [ ] Phase 4 — Preprocessing & feature engineering
- [ ] Phase 5 — Modeling
- [ ] Phase 6 — Evaluation
- [ ] Phase 7 — Model improvement
- [ ] Phase 8 — Explainability
- [ ] Phase 9 — Production pipeline
- [ ] Phase 10 — FastAPI
- [ ] Phase 11 — Streamlit
- [ ] Phase 12 — Testing
- [ ] Phase 13 — Documentation
- [ ] Phase 14 — Deployment preparation
