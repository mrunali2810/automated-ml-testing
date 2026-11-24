"""Tests for training, persistence and data generation (pipeline plumbing).

Persistence tests redirect all file paths to a tmp dir, so they never touch the
real model used by the other tests.
"""
import pandas as pd
import pytest

from src import config, model as model_mod
from src.data_validation import DataValidationError, validate_dataframe
from src.generate_data import generate_dataset

pytestmark = pytest.mark.model


def test_generated_dataset_passes_validation():
    report = validate_dataframe(generate_dataset(n=800, seed=3))
    assert report.is_valid, str(report)


def test_dataset_generation_is_reproducible():
    pd.testing.assert_frame_equal(generate_dataset(n=300, seed=11), generate_dataset(n=300, seed=11))


def test_train_model_returns_metrics_with_expected_keys(raw_df):
    _, metrics = model_mod.train_model(raw_df)
    assert {"accuracy", "f1", "n_train", "n_test", "feature_means", "feature_stds"} <= set(metrics)
    assert metrics["n_train"] + metrics["n_test"] == len(raw_df)
    assert 0.5 < metrics["accuracy"] <= 1.0


def test_training_is_reproducible(raw_df):
    _, m1 = model_mod.train_model(raw_df)
    _, m2 = model_mod.train_model(raw_df)
    assert m1["accuracy"] == m2["accuracy"], "Training is not deterministic"


def test_training_refuses_invalid_data(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "credit_score"] = -10
    with pytest.raises(DataValidationError, match="credit_score"):
        model_mod.train_model(bad)


def test_split_is_stratified_and_disjoint(raw_df):
    X_train, X_test, y_train, y_test = model_mod.split_data(raw_df)
    assert abs(y_train.mean() - y_test.mean()) < 0.01, "Split is not stratified"
    assert set(X_train.index).isdisjoint(X_test.index), "Train/test leakage: shared rows"
    assert len(X_test) == pytest.approx(len(raw_df) * config.TEST_SIZE, abs=1)


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "m" / "model.joblib")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "m" / "metrics.json")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "RAW_DATA_PATH", tmp_path / "raw" / "loan_data.csv")
    return tmp_path


def test_load_or_train_trains_then_loads_identical_model(tmp_paths, X_test):
    assert not config.MODEL_PATH.exists()
    first = model_mod.load_or_train_model()  # trains + saves (dataset auto-generated in tmp)
    assert config.MODEL_PATH.exists() and config.METRICS_PATH.exists()
    second = model_mod.load_or_train_model()  # loads from disk
    assert (model_mod.predict(first, X_test) == model_mod.predict(second, X_test)).all()
    assert model_mod.load_metrics()["accuracy"] > 0.7


def test_main_writes_model_and_processed_splits(tmp_paths, capsys):
    model_mod.main()
    assert config.MODEL_PATH.exists()
    assert (config.PROCESSED_DIR / "train.csv").exists() and (config.PROCESSED_DIR / "test.csv").exists()
    assert "Test accuracy" in capsys.readouterr().out
