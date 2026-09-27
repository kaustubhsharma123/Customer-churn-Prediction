# Customer Churn Prediction System — Portfolio Summary

End-to-end ML system that scores a telecom customer's churn probability, turns it into a retention decision, explains the drivers, and serves it through an API and a web UI. Full details: [README](../README.md).

## Architecture

```
Streamlit UI ──HTTP──► FastAPI ──► churn.predict ──► saved scikit-learn Pipeline (.joblib)
                                   validate input      preprocessing + HistGradientBoosting
```

Preprocessing lives inside the saved Pipeline, so training and serving use identical transformations; the API and UI never re-implement it (enforced by tests).

## Final model

| | |
|---|---|
| Model | HistGradientBoosting (tuned, regularised: 7-leaf trees, ≥ 200 samples/leaf, L2 5.63, learning rate 0.0158, 300 iterations) |
| Decision threshold | 0.30 — highest precision with out-of-fold recall ≥ 0.75 |
| Risk bands | LOW < 0.30 ≤ MEDIUM < 0.60 ≤ HIGH |
| Training data | 5,634 customers (80% stratified split of IBM Telco, 7,043 rows, 26.5% churn) |

## Validated results

| Evaluation | Result |
|---|---|
| Final model, 15-fold repeated CV (training data) | ROC-AUC 0.850 ± 0.012, PR-AUC 0.669; at threshold 0.30: recall 0.760, precision 0.546 |
| Baseline Logistic Regression, single held-out test evaluation (1,409 customers) | ROC-AUC 0.842 [95% CI 0.820–0.862], recall 0.749, precision 0.529 at threshold 0.31 |
| Generalisation (baseline) | Train / CV / test ROC-AUC 0.849 / 0.846 / 0.842 |
| Tuning effect | Tree-model overfitting removed (train–CV gap 0.126 → 0.016); gain over Logistic Regression small (+0.003 ROC-AUC) |

The held-out test set was used once (baseline) and never for tuning; the tuned model's figures are cross-validation estimates.

**Top drivers (SHAP + grouped permutation importance):** contract type, internet service with add-ons, and tenure with charges. Month-to-month contracts, short tenure, fiber optic, and no online security or tech support are associated with higher predicted risk.

## Quality

- **214 automated tests:** data validation, preprocessing, CV, thresholds, tuning, SHAP, artifact integrity, API, UI client, and end-to-end integration on the real artifact.
- Live smoke test (Streamlit → FastAPI → artifact) matches the saved model's own predictions exactly.
- Reproducible from a fresh clone: checksum-verified data, pinned dependencies, and a byte-identical model rebuild.

## Technologies

Python 3.12 · pandas · NumPy · scikit-learn · SciPy · SHAP · Matplotlib/Seaborn · FastAPI · Pydantic · Uvicorn · Streamlit · joblib · pytest

## Deployment

- **API:** Render free web service (`render.yaml`). The model is built from source at deploy time (download → checksum → train); no binaries in git.
- **UI:** Streamlit Community Cloud, configured with `CHURN_API_URL`.

## Key engineering decisions

- One Pipeline for preprocessing and model, to prevent leakage and training/serving skew.
- Decisions written to disk before the single test-set evaluation; tuning used cross-validation only.
- Pre-stated rules for feature, threshold, and model choices; paired fold-level comparisons.
- Threshold treated as a business decision (recall-oriented), not a default of 0.5.
- Fail loudly: strict validation at the API, schema checks, and checksummed artifacts.
