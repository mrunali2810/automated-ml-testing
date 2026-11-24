"""Preprocessing shared by training AND serving.

Using one code path for both is what makes silent train/serve skew detectable:
if this module changes behaviour, the preprocessing tests and the accuracy test
fail.
"""
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def normalize_region(series: pd.Series) -> pd.Series:
    """' North ' -> 'north' (harmless formatting differences must not matter)."""
    return series.astype(str).str.strip().str.lower()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Training-time cleaning: drop exact duplicates and normalise text."""
    out = df.drop_duplicates().copy()
    if "region" in out.columns:
        out["region"] = normalize_region(out["region"])
    return out.reset_index(drop=True)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select model features in the canonical order. Used by training and API."""
    missing: List[str] = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature column(s): {missing}")
    features = df[config.FEATURE_COLUMNS].copy()
    features["region"] = normalize_region(features["region"])
    return features


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features, one-hot encode categoricals (unknowns ignored)."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ]
    )
