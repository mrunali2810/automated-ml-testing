"""Data validation tests.

Pattern: (1) the real dataset must pass; (2) for every check, deliberately
corrupt a copy of the data and assert the validator catches it. Group (2)
proves the validator really detects problems instead of always passing.
"""
import numpy as np
import pytest

from src import config
from src.data_validation import DataValidationError, assert_valid, validate_dataframe

pytestmark = pytest.mark.data


def errors_for(df, **kw):
    return validate_dataframe(df, **kw).errors


def test_real_dataset_passes_all_checks(df):
    report = validate_dataframe(df)
    assert report.is_valid, f"Real dataset failed validation:\n{report}"


def test_dataset_is_not_trivially_small(df):
    assert len(df) >= 1000, f"Dataset has only {len(df)} rows; tests would be meaningless"


def test_both_classes_present(df):
    assert set(df[config.TARGET].unique()) == {0, 1}, "Target must contain both classes"


def test_required_columns_missing_detected(df):
    bad = df.drop(columns=["credit_score"])
    errs = errors_for(bad)
    assert any("Missing required column" in e and "credit_score" in e for e in errs), errs


def test_unexpected_column_detected(df):
    bad = df.assign(leaked_label=df[config.TARGET])
    errs = errors_for(bad)
    assert any("Unexpected column" in e and "leaked_label" in e for e in errs), errs


def test_null_values_detected(df):
    bad = df.copy()
    bad.loc[0:4, "annual_income"] = np.nan
    errs = errors_for(bad)
    assert any("annual_income" in e and "5 null" in e for e in errs), errs


def test_duplicate_rows_detected(df):
    bad = df.iloc[list(range(len(df))) + [0, 1, 2]].reset_index(drop=True)
    errs = errors_for(bad)
    assert any("3 duplicate" in e for e in errs), errs


def test_wrong_dtype_detected(df):
    bad = df.copy()
    bad["age"] = bad["age"].astype(str)  # numbers stored as text
    errs = errors_for(bad)
    assert any("'age' must be numeric" in e for e in errs), errs


def test_fractional_value_in_integer_column_detected(df):
    bad = df.copy()
    bad["credit_score"] = bad["credit_score"].astype(float)
    bad.loc[0, "credit_score"] = 700.5
    errs = errors_for(bad)
    assert any("'credit_score' must contain whole numbers" in e for e in errs), errs


@pytest.mark.parametrize(
    "column, bad_value",
    [
        ("age", 5),
        ("age", 150),
        ("credit_score", 1200),
        ("annual_income", -50.0),
        ("loan_amount", 10_000_000.0),
        ("num_late_payments", -1),
    ],
)
def test_out_of_range_values_detected(df, column, bad_value):
    bad = df.copy()
    bad[column] = bad[column].astype(float)
    bad.loc[0, column] = bad_value
    errs = errors_for(bad)
    assert any(f"'{column}'" in e and "outside" in e for e in errs), errs


def test_invalid_category_detected(df):
    bad = df.copy()
    bad.loc[0, "region"] = "atlantis"
    errs = errors_for(bad)
    assert any("'region'" in e and "atlantis" in e for e in errs), errs


def test_invalid_target_value_detected(df):
    bad = df.copy()
    bad.loc[0, config.TARGET] = 2
    errs = errors_for(bad)
    assert any("must be 0/1" in e for e in errs), errs


def test_business_rule_employment_vs_age_detected(df):
    bad = df.copy()
    bad.loc[0, "age"] = 20
    bad.loc[0, "employment_years"] = 30
    errs = errors_for(bad)
    assert any("employment_years > age" in e for e in errs), errs


def test_feature_only_validation_does_not_require_target(df):
    features_only = df.drop(columns=[config.TARGET])
    assert validate_dataframe(features_only, include_target=False).is_valid
    # ...but the same frame must fail when the target IS required.
    assert not validate_dataframe(features_only, include_target=True).is_valid


def test_assert_valid_raises_with_readable_message(df):
    bad = df.drop(columns=["age"])
    with pytest.raises(DataValidationError, match="Missing required column"):
        assert_valid(bad)


def test_assert_valid_silent_on_good_data(df):
    assert_valid(df)  # must not raise


def test_multiple_problems_reported_together(df):
    bad = df.copy()
    bad.loc[0, "region"] = "mars"
    bad.loc[1, "annual_income"] = np.nan
    errs = errors_for(bad)
    assert len(errs) >= 2, f"Expected both problems to be reported, got: {errs}"


def test_numeric_values_in_text_column_detected(df):
    bad = df.copy()
    bad["region"] = 1  # region stored as numbers instead of text
    errs = errors_for(bad)
    assert any("'region' must be text" in e for e in errs), errs


def test_non_numeric_target_detected(df):
    bad = df.copy()
    bad[config.TARGET] = bad[config.TARGET].map({0: "no", 1: "yes"})
    errs = errors_for(bad)
    assert any(f"Target '{config.TARGET}' must be numeric" in e for e in errs), errs
