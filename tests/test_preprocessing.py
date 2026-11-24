"""Preprocessing tests: guard against silent train/serve skew.

These tests would have caught the 'silent data bug' from the assignment
scenario: they check that serving-time preprocessing leaves the data
*numerically unchanged* and statistically consistent with training data.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.preprocessing import build_preprocessor, clean_data, normalize_region, prepare_features

pytestmark = pytest.mark.data


def test_prepare_features_returns_canonical_column_order(X_test):
    out = prepare_features(X_test.sample(frac=1, axis=1, random_state=1))  # shuffled columns
    assert list(out.columns) == config.FEATURE_COLUMNS


def test_prepare_features_does_not_change_numeric_values(X_test):
    """Numeric feature values must pass through untouched (catches scaling bugs)."""
    out = prepare_features(X_test)
    for col in config.NUMERIC_FEATURES:
        np.testing.assert_allclose(
            out[col].to_numpy(dtype=float),
            X_test[col].to_numpy(dtype=float),
            err_msg=f"prepare_features altered column '{col}' (scaling / unit / swap bug?)",
        )


def test_prepare_features_does_not_mutate_input(X_test):
    before = X_test.copy()
    prepare_features(X_test)
    pd.testing.assert_frame_equal(X_test, before)


def test_prepare_features_raises_on_missing_column(X_test):
    with pytest.raises(KeyError, match="credit_score"):
        prepare_features(X_test.drop(columns=["credit_score"]))


def test_normalize_region_handles_case_and_whitespace():
    out = normalize_region(pd.Series([" North ", "SOUTH", "east", "\tWest\n"]))
    assert out.tolist() == ["north", "south", "east", "west"]


def test_clean_data_removes_duplicates_and_normalises(raw_df):
    dirty = pd.concat([raw_df.head(10), raw_df.head(3)], ignore_index=True)
    dirty["region"] = dirty["region"].str.upper()
    cleaned = clean_data(dirty)
    assert len(cleaned) == 10, "Duplicates were not removed"
    assert set(cleaned["region"]) <= set(config.CATEGORICAL_VALUES["region"])


def test_serving_features_match_training_distribution(X_test, metrics):
    """Train/serve skew check: mean of each served feature must sit within
    0.15 training-std of the training mean. With ~1000 test rows the sampling
    error is ~0.03 std, so 0.15 leaves wide margin yet catches unit/scale bugs
    (e.g. income divided by 1000 would be ~2 std away)."""
    served = prepare_features(X_test)[config.NUMERIC_FEATURES]
    for col in config.NUMERIC_FEATURES:
        shift = abs(served[col].mean() - metrics["feature_means"][col]) / metrics["feature_stds"][col]
        assert shift < 0.15, (
            f"Feature '{col}' distribution shifted by {shift:.2f} training-stds between "
            f"training and serving (served mean={served[col].mean():.2f}, "
            f"training mean={metrics['feature_means'][col]:.2f})"
        )


def test_preprocessor_output_is_scaled_and_finite(X_test, raw_df):
    X_train = raw_df.drop(columns=[config.TARGET])
    pre = build_preprocessor().fit(prepare_features(X_train))
    out = pre.transform(prepare_features(X_test))
    out = out.toarray() if hasattr(out, "toarray") else np.asarray(out)
    assert np.isfinite(out).all(), "Preprocessor produced NaN/inf"
    n_num = len(config.NUMERIC_FEATURES)
    assert np.abs(out[:, :n_num].mean(axis=0)).max() < 0.2, "Numeric features are not centred"
    assert out.shape[1] == n_num + len(config.CATEGORICAL_VALUES["region"])
