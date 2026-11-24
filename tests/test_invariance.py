"""B. Invariance tests: harmless input changes must not change predictions.

Perturbations (all semantically irrelevant to a loan decision here):
  1. `region` is not a driver of the label -> switching it must barely move probability.
     Tolerances are calibrated on the measured model: it learns small noise
     coefficients for region (|coef| <= ~0.13 logits => max shift ~0.06). We allow
     max 0.10 / mean 0.03; a genuine region dependence (coef ~1) would shift
     probabilities by >= 0.25 and fail.
  2. Tiny relative noise (0.1 %) on income / loan amount.
  3. Row order of the batch.
  4. Text formatting of `region` (case / whitespace).
Probabilities are compared with tolerances, not only labels, so the test is
sensitive but not flaky.
"""
import numpy as np
import pytest

from src import config
from src.model import predict, predict_proba

pytestmark = pytest.mark.model


@pytest.mark.parametrize("new_region", config.CATEGORICAL_VALUES["region"])
def test_region_change_does_not_move_probability(model, sample_rows, new_region):
    base = predict_proba(model, sample_rows)[:, 1]
    changed = predict_proba(model, sample_rows.assign(region=new_region))[:, 1]
    shift = np.abs(base - changed)
    assert shift.max() < 0.10, f"Switching region to '{new_region}' moved a probability by {shift.max():.3f} (> 0.10)"
    assert shift.mean() < 0.03, f"Mean probability shift {shift.mean():.3f} from region change (> 0.03)"


def test_region_change_rarely_flips_labels(model, sample_rows):
    base = predict(model, sample_rows)
    flips = 0
    for region in config.CATEGORICAL_VALUES["region"]:
        flips += int((predict(model, sample_rows.assign(region=region)) != base).sum())
    rate = flips / (len(sample_rows) * len(config.CATEGORICAL_VALUES["region"]))
    assert rate <= 0.02, f"{rate:.1%} of predictions flipped just because of region (limit 2%)"


def test_tiny_numeric_noise_does_not_change_predictions(model, sample_rows):
    noisy = sample_rows.copy()
    noisy["annual_income"] *= 1.001
    noisy["loan_amount"] *= 0.999
    base = predict_proba(model, sample_rows)[:, 1]
    after = predict_proba(model, noisy)[:, 1]
    assert np.abs(base - after).max() < 0.01, "0.1% noise changed probabilities by more than 0.01"
    assert (predict(model, sample_rows) == predict(model, noisy)).mean() >= 0.99, (
        "More than 1% of labels flipped under 0.1% input noise"
    )


def test_prediction_independent_of_row_order(model, sample_rows):
    shuffled = sample_rows.sample(frac=1, random_state=7)
    p_orig = pd_series(predict_proba(model, sample_rows)[:, 1], sample_rows.index)
    p_shuf = pd_series(predict_proba(model, shuffled)[:, 1], shuffled.index)
    np.testing.assert_allclose(p_orig.sort_index().to_numpy(), p_shuf.sort_index().to_numpy(), atol=1e-12,
                               err_msg="Predictions depend on row order within a batch")


def test_region_text_formatting_does_not_change_predictions(model, sample_rows):
    messy = sample_rows.assign(region=sample_rows["region"].str.upper().map(lambda r: f"  {r}\t"))
    np.testing.assert_array_equal(predict(model, sample_rows), predict(model, messy),
                                  err_msg="Case/whitespace in 'region' changed predictions")


def test_column_order_of_input_does_not_matter(model, sample_rows):
    reordered = sample_rows[list(reversed(sample_rows.columns))]
    np.testing.assert_allclose(predict_proba(model, sample_rows), predict_proba(model, reordered),
                               atol=1e-12, err_msg="Input column order changed the prediction")


def pd_series(values, index):
    import pandas as pd
    return pd.Series(values, index=index)
