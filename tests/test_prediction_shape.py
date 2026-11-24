"""D. Prediction shape / type / sanity tests."""
import numpy as np
import pytest

from src.model import predict, predict_proba

pytestmark = pytest.mark.model


def test_prediction_count_matches_input_count(model, X_test):
    assert len(predict(model, X_test)) == len(X_test)


def test_prediction_is_one_dimensional(model, X_test):
    preds = predict(model, X_test)
    assert preds.ndim == 1 and preds.shape == (len(X_test),)


def test_predictions_are_binary_integers(model, X_test):
    preds = predict(model, X_test)
    assert np.issubdtype(preds.dtype, np.integer), f"Expected integer labels, got {preds.dtype}"
    assert set(np.unique(preds)) <= {0, 1}


def test_predictions_have_no_nan_or_inf(model, X_test):
    preds = predict(model, X_test).astype(float)
    assert np.isfinite(preds).all(), "Predictions contain NaN or inf"


def test_probability_matrix_shape_and_values(model, X_test):
    proba = predict_proba(model, X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.isfinite(proba).all(), "Probabilities contain NaN or inf"
    assert ((proba >= 0) & (proba <= 1)).all(), "Probabilities outside [0, 1]"
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9, err_msg="Probabilities do not sum to 1")


def test_single_sample_prediction(model, X_test):
    preds = predict(model, X_test.iloc[[0]])
    assert preds.shape == (1,)


@pytest.mark.parametrize("n", [1, 7, 100])
def test_output_length_scales_with_batch_size(model, X_test, n):
    assert len(predict(model, X_test.head(n))) == n


def test_labels_consistent_with_probabilities(model, X_test):
    proba = predict_proba(model, X_test)[:, 1]
    np.testing.assert_array_equal(predict(model, X_test), (proba >= 0.5).astype(int))


def test_prediction_is_deterministic(model, X_test):
    np.testing.assert_array_equal(predict(model, X_test), predict(model, X_test))
