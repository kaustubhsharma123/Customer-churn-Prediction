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
