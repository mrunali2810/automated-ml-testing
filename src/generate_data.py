"""Generate a reproducible synthetic loan-approval dataset.

Run:  python -m src.generate_data
The approval label depends on credit score, income, loan size, late payments
and employment history. `region` and `age` are deliberately NOT drivers of the
label, which lets the invariance tests assert that they do not matter.
"""
import numpy as np
import pandas as pd

from src import config


def generate_dataset(n: int = config.N_SAMPLES, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lo = lambda c: config.NUMERIC_RANGES[c][0]  # noqa: E731
    hi = lambda c: config.NUMERIC_RANGES[c][1]  # noqa: E731

    age = rng.integers(21, 70, n)
    max_years = np.minimum(age - config.MIN_WORKING_AGE - 4, hi("employment_years"))
    employment_years = np.minimum(rng.integers(0, 40, n), max_years)
    annual_income = np.clip(rng.lognormal(10.9, 0.45, n), lo("annual_income"), hi("annual_income")).round(2)
    credit_score = np.clip(rng.normal(680, 70, n), lo("credit_score"), hi("credit_score")).round().astype(int)
    loan_amount = np.clip(rng.lognormal(9.8, 0.7, n), lo("loan_amount"), hi("loan_amount")).round(2)
    late = np.clip(rng.poisson(1.2, n), 0, hi("num_late_payments"))
    region = rng.choice(config.CATEGORICAL_VALUES["region"], n)

    z = lambda x: (x - x.mean()) / x.std()  # noqa: E731
    logit = (
        1.7 * z(credit_score)
        + 1.0 * z(annual_income)
        - 1.2 * z(loan_amount)
        - 0.9 * z(late)
        + 0.4 * z(employment_years)
        + rng.normal(0, 0.3, n)
    )
    approved = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "employment_years": employment_years,
            "num_late_payments": late,
            "region": region,
            config.TARGET: approved,
        }
    )
    return df.drop_duplicates().reset_index(drop=True)


def main() -> None:
    df = generate_dataset()
    config.RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_DATA_PATH, index=False)
    print(f"Wrote {len(df)} rows to {config.RAW_DATA_PATH} (approval rate {df[config.TARGET].mean():.2%})")


if __name__ == "__main__":
    main()
