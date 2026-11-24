"""Data validation: schema, nulls, duplicates, dtypes, ranges, invalid values.

Every check returns a list of human-readable error strings (empty == OK) so a
test failure message tells you exactly *what* is wrong with the data.
"""
from dataclasses import dataclass, field
from typing import List

import pandas as pd
from pandas.api import types as pdt

from src import config


class DataValidationError(ValueError):
    """Raised by `assert_valid` when a DataFrame violates the data contract."""


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        return "OK" if self.is_valid else "\n".join(f"- {e}" for e in self.errors)


def _expected_columns(include_target: bool) -> List[str]:
    return config.REQUIRED_COLUMNS if include_target else config.FEATURE_COLUMNS


def check_required_columns(df: pd.DataFrame, include_target: bool = True) -> List[str]:
    missing = [c for c in _expected_columns(include_target) if c not in df.columns]
    return [f"Missing required column(s): {missing}"] if missing else []


def check_unexpected_columns(df: pd.DataFrame, include_target: bool = True) -> List[str]:
    extra = [c for c in df.columns if c not in _expected_columns(include_target)]
    return [f"Unexpected column(s): {extra}"] if extra else []


def check_nulls(df: pd.DataFrame, include_target: bool = True) -> List[str]:
    errors = []
    for col in _expected_columns(include_target):
        if col in df.columns and df[col].isna().any():
            errors.append(f"Column '{col}' has {int(df[col].isna().sum())} null value(s)")
    return errors


def check_duplicates(df: pd.DataFrame) -> List[str]:
    n = int(df.duplicated().sum())
    return [f"Found {n} duplicate row(s)"] if n else []


def check_dtypes(df: pd.DataFrame, include_target: bool = True) -> List[str]:
    errors = []
    for col in config.NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        s = df[col]
        if pdt.is_bool_dtype(s) or not pdt.is_numeric_dtype(s):
            errors.append(f"Column '{col}' must be numeric, got dtype {s.dtype}")
        elif col in config.INTEGER_COLUMNS:
            non_null = s.dropna()
            if not (non_null == non_null.round()).all():
                errors.append(f"Column '{col}' must contain whole numbers")
    for col in config.CATEGORICAL_FEATURES:
        if col in df.columns and not (pdt.is_string_dtype(df[col]) or pdt.is_object_dtype(df[col])):
            errors.append(f"Column '{col}' must be text, got dtype {df[col].dtype}")
    if include_target and config.TARGET in df.columns:
        t = df[config.TARGET]
        if pdt.is_bool_dtype(t) or not pdt.is_numeric_dtype(t):
            errors.append(f"Target '{config.TARGET}' must be numeric 0/1, got dtype {t.dtype}")
    return errors


def check_ranges(df: pd.DataFrame) -> List[str]:
    errors = []
    for col, (lo, hi) in config.NUMERIC_RANGES.items():
        if col not in df.columns or not pdt.is_numeric_dtype(df[col]):
            continue
        bad = df[col].dropna()
        bad = bad[(bad < lo) | (bad > hi)]
        if len(bad):
            errors.append(
                f"Column '{col}' has {len(bad)} value(s) outside [{lo}, {hi}] "
                f"(observed min={bad.min()}, max={bad.max()})"
            )
    return errors


def check_categories(df: pd.DataFrame) -> List[str]:
    errors = []
    for col, allowed in config.CATEGORICAL_VALUES.items():
        if col in df.columns:
            bad = set(df[col].dropna().astype(str)) - set(allowed)
            if bad:
                errors.append(f"Column '{col}' has invalid value(s) {sorted(bad)}; allowed: {allowed}")
    return errors


def check_target_values(df: pd.DataFrame) -> List[str]:
    if config.TARGET not in df.columns:
        return []
    bad = set(df[config.TARGET].dropna().unique()) - {0, 1}
    return [f"Target '{config.TARGET}' must be 0/1, found {sorted(bad)}"] if bad else []


def check_business_rules(df: pd.DataFrame) -> List[str]:
    """Cross-column plausibility: employment cannot exceed (age - 14)."""
    cols = ["age", "employment_years"]
    if not set(cols) <= set(df.columns) or not all(pdt.is_numeric_dtype(df[c]) for c in cols):
        return []  # missing / non-numeric columns are already reported by other checks
    n = int((df["employment_years"] > df["age"] - config.MIN_WORKING_AGE).sum())
    return [f"{n} row(s) have employment_years > age - {config.MIN_WORKING_AGE}"] if n else []


def validate_dataframe(df: pd.DataFrame, include_target: bool = True) -> ValidationReport:
    """Run every check and collect all errors (does not stop at the first)."""
    errors: List[str] = []
    errors += check_required_columns(df, include_target)
    errors += check_unexpected_columns(df, include_target)
    errors += check_nulls(df, include_target)
    errors += check_duplicates(df)
    errors += check_dtypes(df, include_target)
    errors += check_ranges(df)
    errors += check_categories(df)
    if include_target:
        errors += check_target_values(df)
    errors += check_business_rules(df)
    return ValidationReport(errors)


def assert_valid(df: pd.DataFrame, include_target: bool = True) -> None:
    report = validate_dataframe(df, include_target)
    if not report.is_valid:
        raise DataValidationError(f"Data validation failed:\n{report}")
