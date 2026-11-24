"""Shared, reusable fixtures (session-scoped so the model is loaded once)."""
import copy

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import config
from src.api import app
from src.model import load_dataset, load_metrics, load_or_train_model, split_data
from src.preprocessing import clean_data


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """The dataset exactly as stored on disk (what validation must approve)."""
    return load_dataset()


@pytest.fixture(scope="session")
def splits(raw_df):
    """(X_train, X_test, y_train, y_test) using the same split as training."""
    return split_data(clean_data(raw_df))


@pytest.fixture(scope="session")
def model():
    return load_or_train_model()


@pytest.fixture(scope="session")
def metrics():
    """Metrics recorded at training time (baseline accuracy, feature stats)."""
    load_or_train_model()  # guarantees metrics.json exists
    return load_metrics()


@pytest.fixture(scope="session")
def X_test(splits):
    return splits[1]


@pytest.fixture(scope="session")
def y_test(splits):
    return splits[3]


@pytest.fixture
def df(raw_df) -> pd.DataFrame:
    """A fresh copy per test, so tests can corrupt it freely."""
    return raw_df.copy()


@pytest.fixture
def sample_rows(X_test) -> pd.DataFrame:
    """200 reproducible test rows for invariance / directional checks."""
    return X_test.sample(200, random_state=config.RANDOM_STATE).reset_index(drop=True)


@pytest.fixture
def valid_payload() -> dict:
    return {
        "age": 35,
        "annual_income": 60000.0,
        "credit_score": 700,
        "loan_amount": 15000.0,
        "employment_years": 8,
        "num_late_payments": 0,
        "region": "north",
    }


@pytest.fixture
def payload(valid_payload):
    """Deep copy that tests may mutate."""
    return copy.deepcopy(valid_payload)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
