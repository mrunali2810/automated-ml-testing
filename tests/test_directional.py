"""C. Directional tests: moving an important feature in a known direction must
move the approval probability in the expected direction.

ASSUMPTIONS (domain knowledge, documented as required):
  * credit_score      UP    -> approval probability must NOT decrease
  * annual_income     UP    -> approval probability must NOT decrease
  * employment_years  UP    -> approval probability must NOT decrease
  * loan_amount       UP    -> approval probability must NOT increase
  * num_late_payments UP    -> approval probability must NOT increase
Values are clipped to the valid range after the shift. Because the model is a
logistic regression whose learned coefficients have the expected sign, the
relationship is monotone for *every* row, so we assert zero violations plus a
meaningful average effect (so a flat/ignored feature also fails).
"""
import pytest

from src import config
from src.model import predict_proba

pytestmark = pytest.mark.model

# feature, shift, expected sign of probability change
CASES = [
    ("credit_score", +100, +1),
    ("annual_income", +30000.0, +1),
    ("employment_years", +5, +1),
    ("loan_amount", +20000.0, -1),
    ("num_late_payments", +3, -1),
]


def shifted(df, feature, delta):
    lo, hi = config.NUMERIC_RANGES[feature]
    out = df.copy()
    out[feature] = (out[feature] + delta).clip(lo, hi)
    return out


@pytest.mark.parametrize("feature, delta, sign", CASES)
def test_feature_moves_probability_in_expected_direction(model, sample_rows, feature, delta, sign):
    base = predict_proba(model, sample_rows)[:, 1]
    after = predict_proba(model, shifted(sample_rows, feature, delta))[:, 1]
    diff = sign * (after - base)  # positive == moved the expected way

    violations = int((diff < -1e-9).sum())
    assert violations == 0, (
        f"{violations}/{len(diff)} rows moved the WRONG way when '{feature}' changed by {delta:+}"
    )
    assert diff.mean() > 0.005, (
        f"'{feature}' has almost no effect (mean expected-direction change {diff.mean():.4f}); "
        f"the feature may be ignored or corrupted"
    )


def test_obviously_good_applicant_beats_obviously_bad(model, valid_payload):
    import pandas as pd

    good = {**valid_payload, "credit_score": 820, "annual_income": 150000.0, "loan_amount": 5000.0,
            "num_late_payments": 0, "employment_years": 20}
    bad = {**valid_payload, "credit_score": 400, "annual_income": 20000.0, "loan_amount": 100000.0,
           "num_late_payments": 12, "employment_years": 1}
    p = predict_proba(model, pd.DataFrame([good, bad]))[:, 1]
    assert p[0] > 0.9 and p[1] < 0.1, f"Good applicant p={p[0]:.3f}, bad applicant p={p[1]:.3f}"
