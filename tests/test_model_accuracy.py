"""A. Minimum accuracy tests.

Threshold rationale (see README): the baseline measured at training time is
~0.81 and the majority-class baseline is ~0.50.
 * MIN_ACCURACY (default 0.75, env-configurable) is an absolute floor, far above chance.
 * A second test fails if accuracy falls >0.03 below the recorded baseline, which
   catches slow degradation that the absolute floor alone would miss.
"""
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src import config
from src.model import predict

pytestmark = pytest.mark.model


def test_accuracy_meets_minimum_threshold(model, X_test, y_test):
    acc = accuracy_score(y_test, predict(model, X_test))
    assert acc >= config.MIN_ACCURACY, (
        f"Model accuracy {acc:.4f} is below the minimum {config.MIN_ACCURACY:.2f} "
        f"(set MIN_ACCURACY to change). Possible data or preprocessing bug."
    )


def test_accuracy_has_not_regressed_vs_training_baseline(model, X_test, y_test, metrics):
    acc = accuracy_score(y_test, predict(model, X_test))
    floor = metrics["accuracy"] - config.MAX_ACCURACY_DROP_VS_BASELINE
    assert acc >= floor, (
        f"Accuracy {acc:.4f} dropped more than {config.MAX_ACCURACY_DROP_VS_BASELINE} below the "
        f"training baseline {metrics['accuracy']:.4f}: model or serving pipeline has degraded."
    )


def test_model_beats_majority_class_baseline(model, X_test, y_test, metrics):
    acc = accuracy_score(y_test, predict(model, X_test))
    assert acc > metrics["majority_class_baseline"] + 0.10, (
        f"Accuracy {acc:.4f} is barely better than always predicting the majority class "
        f"({metrics['majority_class_baseline']:.4f}); the model has learned little."
    )


def test_f1_score_is_reasonable(model, X_test, y_test):
    f1 = f1_score(y_test, predict(model, X_test))
    assert f1 >= 0.70, f"F1 {f1:.4f} < 0.70 - model is unbalanced or broken"


def test_model_predicts_both_classes(model, X_test):
    preds = predict(model, X_test)
    assert set(preds) == {0, 1}, f"Model collapsed to a single class: {set(preds)}"
