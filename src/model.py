"""Model training, persistence and inference helpers."""
import json
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.data_validation import assert_valid
from src.preprocessing import build_preprocessor, clean_data, prepare_features


def load_dataset(path=None) -> pd.DataFrame:
    path = config.RAW_DATA_PATH if path is None else path  # resolved at call time
    if not path.exists():
        from src.generate_data import main as generate  # lazily create the dataset

        generate()
    return pd.read_csv(path)


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Deterministic stratified split, so tests and training see the same test set."""
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )


def build_model() -> Pipeline:
    # Logistic regression: fast, deterministic and monotonic in each feature,
    # which makes the directional tests meaningful and stable.
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)),
        ]
    )


def train_model(df: pd.DataFrame = None) -> Tuple[Pipeline, Dict]:
    """Validate -> clean -> split -> fit -> evaluate. Returns (model, metrics)."""
    df = load_dataset() if df is None else df
    assert_valid(df)  # refuse to train on bad data
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    model = build_model()
    model.fit(prepare_features(X_train), y_train)
    preds = model.predict(prepare_features(X_test))

    train_feats = prepare_features(X_train)[config.NUMERIC_FEATURES]
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "majority_class_baseline": float(max(y_test.mean(), 1 - y_test.mean())),
        # Reference statistics used by the train/serve skew test.
        "feature_means": train_feats.mean().to_dict(),
        "feature_stds": train_feats.std().to_dict(),
    }
    return model, metrics


def save_model(model: Pipeline, metrics: Dict) -> None:
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))


def load_metrics() -> Dict:
    return json.loads(config.METRICS_PATH.read_text())


def load_or_train_model() -> Pipeline:
    """Load the saved model; train and save it first if it does not exist."""
    if config.MODEL_PATH.exists() and config.METRICS_PATH.exists():
        return joblib.load(config.MODEL_PATH)
    model, metrics = train_model()
    save_model(model, metrics)
    return model


def predict(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return model.predict(prepare_features(df))


def predict_proba(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare_features(df))


def main() -> None:
    model, metrics = train_model()
    save_model(model, metrics)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_data(load_dataset())
    X_train, X_test, y_train, y_test = split_data(df)
    X_train.assign(**{config.TARGET: y_train}).to_csv(config.PROCESSED_DIR / "train.csv", index=False)
    X_test.assign(**{config.TARGET: y_test}).to_csv(config.PROCESSED_DIR / "test.csv", index=False)
    print(f"Trained model. Test accuracy={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} "
          f"(majority-class baseline={metrics['majority_class_baseline']:.4f})")
    print(f"Saved model to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
