"""Central configuration: paths, schema contract and test thresholds.

The SCHEMA below is the *data contract* shared by the data generator, the
validator, the model and the API, so a change here is picked up everywhere.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- Paths -----------------------------------------------------------------
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "loan_data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

# ---- Reproducibility -------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SAMPLES = 5000

# ---- Data contract ---------------------------------------------------------
TARGET = "approved"

# column -> (min, max) inclusive. Used by validator, API and test fixtures.
NUMERIC_RANGES = {
    "age": (18, 75),
    "annual_income": (10_000.0, 300_000.0),
    "credit_score": (300, 850),
    "loan_amount": (500.0, 150_000.0),
    "employment_years": (0, 50),
    "num_late_payments": (0, 20),
}
INTEGER_COLUMNS = ["age", "credit_score", "employment_years", "num_late_payments"]
FLOAT_COLUMNS = ["annual_income", "loan_amount"]
NUMERIC_FEATURES = INTEGER_COLUMNS + FLOAT_COLUMNS

CATEGORICAL_VALUES = {"region": ["north", "south", "east", "west"]}
CATEGORICAL_FEATURES = list(CATEGORICAL_VALUES)

# Order matters: this is the exact column order the model is served with.
FEATURE_COLUMNS = [
    "age",
    "annual_income",
    "credit_score",
    "loan_amount",
    "employment_years",
    "num_late_payments",
    "region",
]
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET]

# Business rule: nobody can have worked longer than (age - MIN_WORKING_AGE) years.
MIN_WORKING_AGE = 14

# ---- Test thresholds -------------------------------------------------------
# Absolute floor for model accuracy. Override in CI with MIN_ACCURACY=0.85 etc.
# The default was chosen from the measured baseline (see models/metrics.json
# and README "How the accuracy threshold was chosen").
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.75"))

# Maximum accuracy drop tolerated versus the baseline recorded at training time.
MAX_ACCURACY_DROP_VS_BASELINE = float(os.getenv("MAX_ACCURACY_DROP", "0.03"))
