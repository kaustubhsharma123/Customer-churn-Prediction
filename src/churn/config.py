"""Central project configuration: paths and reproducibility settings.

Paths are resolved relative to this file, not the current working directory,
so they are correct whether code runs from a notebook, the API, or pytest.
"""

from pathlib import Path

# src/churn/config.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_FILE = RAW_DATA_DIR / "Telco-Customer-Churn.csv"

# Single seed used for every split and model so results are reproducible.
RANDOM_STATE = 42
