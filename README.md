# PI11 – Automated Testing for Machine Learning

**Scenario:** a silent data bug lowered production model accuracy for weeks without anyone noticing.
**This project** is a loan-approval classifier with an automated test framework that catches data-quality
problems, model failures and API errors **before deployment**, plus a demonstration that it really does catch
an injected silent bug.

> No existing repository was provided, so (as the brief allows) a small reproducible classification project
> was created: synthetic loan data → scikit-learn pipeline → FastAPI service.

## Results (measured, not estimated)

| Item | Result |
|---|---|
| Automated tests | **134 passed** (`reports/test_runs/full_suite_passed.log`) |
| Coverage of `src/` | **99.31 %** (gate: ≥ 80 %); only the two `main()` entry-point lines are uncovered |
| Model baseline (test set, 1 000 rows) | accuracy **0.809**, F1 0.810 (majority-class baseline 0.504) |
| Injected bug "scaling" | **6 tests fail** with the bug → **all 134 pass** after the fix |
| Injected bug "swap" | **15 tests fail** with the bug → **all 134 pass** after the fix |

Verified in a clean room: fresh virtualenv from `requirements.txt`, no pre-trained model, same results.

## Project structure

```text
automated-ml-testing/
├── data/raw/loan_data.csv        # synthetic dataset (reproducible, seed 42)
├── data/processed/               # train.csv / test.csv written by `python -m src.model`
├── src/
│   ├── config.py                 # paths, data contract (ranges, dtypes), thresholds
│   ├── generate_data.py          # reproducible dataset generator
│   ├── data_validation.py        # schema / null / duplicate / dtype / range / category checks
│   ├── preprocessing.py          # cleaning + feature prep shared by training AND serving
│   ├── model.py                  # train, evaluate, save/load, predict
│   └── api.py                    # FastAPI: /health, /predict, /predict/batch
├── tests/
│   ├── conftest.py               # reusable fixtures
│   ├── test_data_validation.py   # 24 tests  (marker: data)
│   ├── test_preprocessing.py     #  8 tests  (marker: data)  train/serve skew guards
│   ├── test_model_accuracy.py    #  5 tests  (marker: model)
│   ├── test_invariance.py        #  9 tests  (marker: model)
│   ├── test_directional.py       #  6 tests  (marker: model)
│   ├── test_prediction_shape.py  # 11 tests  (marker: model)
│   ├── test_model_training.py    #  8 tests  (marker: model) training / persistence plumbing
│   └── test_api.py               # 63 tests  (marker: api)
├── bug_demo/run_bug_demo.py      # injected-bug demonstration (isolated, works on a temp copy)
├── .github/workflows/tests.yml   # CI pipeline
├── reports/coverage/             # HTML coverage report (generated)
├── reports/bug_demo/             # logs of failing + fixed runs (generated)
├── reports/test_runs/            # log of the passing full run
├── pytest.ini  requirements.txt  .gitignore  README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.model                  # generates data if missing, trains and saves the model
```
(The tests also train the model automatically if `models/model.joblib` is missing.)

## Commands

| Task | Command |
|---|---|
| **Run everything** (tests + coverage + 80 % gate) | `pytest` |
| Data validation tests only | `pytest tests/test_data_validation.py tests/test_preprocessing.py --no-cov` |
| Accuracy tests | `pytest tests/test_model_accuracy.py --no-cov` |
| Invariance tests | `pytest tests/test_invariance.py --no-cov` |
| Directional tests | `pytest tests/test_directional.py --no-cov` |
| Prediction shape tests | `pytest tests/test_prediction_shape.py --no-cov` |
| API tests | `pytest tests/test_api.py --no-cov` |
| By group (markers) | `pytest -m data --no-cov`, `pytest -m model --no-cov`, `pytest -m api --no-cov` |
| Coverage report (terminal + HTML) | `pytest` (configured in `pytest.ini`) → `reports/coverage/index.html` |
| Open the HTML report | `python -c "import webbrowser,pathlib;webbrowser.open(pathlib.Path('reports/coverage/index.html').resolve().as_uri())"` (or `open` / `xdg-open` / `start` the file) |
| Run the API | `uvicorn src.api:app --reload` → docs at <http://127.0.0.1:8000/docs> |
| **Injected-bug demo** | `python bug_demo/run_bug_demo.py` (`--bug scaling` default, `--bug swap`, `--bug all`) |
| Change the accuracy floor | `MIN_ACCURACY=0.80 pytest tests/test_model_accuracy.py --no-cov` (Windows PowerShell: `$env:MIN_ACCURACY=0.80`) |

`--no-cov` skips the coverage gate, which only makes sense for the full run (a single module naturally covers
little of `src/`).

Example API call:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"age":35,"annual_income":60000,"credit_score":700,"loan_amount":15000,"employment_years":8,"num_late_payments":0,"region":"north"}'
# {"prediction":1,"approved":true,"probability":0.842899}
```

## What is tested

**Data validation** (`src/data_validation.py`): required columns, unexpected columns, nulls, duplicates, dtypes
(incl. whole numbers in integer columns), numeric ranges, allowed categories, binary target, and a cross-column
business rule (employment years ≤ age − 14). Each check has a test that *corrupts a copy of the data* and
asserts the error is reported, so the validator cannot pass vacuously. Training refuses invalid data.

**Preprocessing** (`test_preprocessing.py`): serving-time preprocessing must leave numeric values unchanged, keep the
canonical column order, not mutate its input, and produce feature distributions consistent with the training
statistics saved in `models/metrics.json`. These tests target exactly the "silent bug" in the scenario.

**Model**
* *Minimum accuracy* – absolute floor `MIN_ACCURACY` (default 0.75, env-configurable), plus a regression test
  (fails if accuracy is > 0.03 below the recorded training baseline), beats-majority-class, F1, both classes predicted.
* *Invariance* – changing `region` (not a label driver), 0.1 % noise on income/loan, row order, column order and
  case/whitespace of `region` must not change predictions.
* *Directional* – assumptions documented in `test_directional.py`: ↑credit score, ↑income, ↑employment years
  must not lower approval probability; ↑loan amount, ↑late payments must not raise it. Every row must obey this,
  and the average effect must be non-trivial (so an ignored/corrupted feature also fails).
* *Prediction shape* – count = input count, 1-D output, integer 0/1 labels, probabilities in [0, 1] summing to 1,
  no NaN/inf, single-row and batch sizes, determinism.

**API** (FastAPI `TestClient`): valid request and response structure; each of the 7 fields missing, null, wrong
type (strings, floats for ints, booleans, lists); out-of-range values; invalid category; NaN/Infinity; unknown
extra field; empty object, no body, invalid JSON; 404/405; batch endpoint; and server-side failures (500 without
leaking internals, 503 when the model is unavailable). Invalid input always yields **422**.

## How the thresholds were chosen (not hard-coded blindly)

* Measured baseline: accuracy 0.809 vs. 0.504 majority-class accuracy. Labels are sampled stochastically from a
  logistic score, so even a perfect model cannot reach 100 %.
* `MIN_ACCURACY = 0.75` sits ~6 points under the baseline (room for sampling noise, well above chance).
* The baseline-regression test (−0.03) is tighter and reacts to smaller degradations.
* Region-invariance tolerance (max 0.10 / mean 0.03): the model fits tiny noise coefficients on `region`
  (|coef| ≤ ≈0.13 → max measured shift ≈0.06). A genuine region dependence would shift probabilities by ≥ 0.25.
* Skew tolerance (0.15 training-stds): sampling error of the test-set mean is ≈0.03 std; an income unit bug is ≈2 std.

## Injected bug demonstration

`python bug_demo/run_bug_demo.py --bug all` copies the project to a temp folder, injects a bug into **the copy's**
`src/preprocessing.py` (production code is never modified), runs the suite (expected: fail), restores the
original and runs again (expected: pass). Both bugs are silent: nothing crashes and the model keeps returning
predictions.

| Bug | Effect | Result with bug | After fix |
|---|---|---|---|
| `scaling` – income divided by 1000 at serving time | accuracy 0.809 → **0.663** | **6 failed**, 128 passed | 134 passed |
| `swap` – credit_score and income values swapped | accuracy → **0.504** (chance) | **15 failed**, 119 passed | 134 passed |

Tests that caught the `scaling` bug (see `reports/bug_demo/scaling_1_buggy_run.log` for the exact messages):

```
Model accuracy 0.6630 is below the minimum 0.75 ... Possible data or preprocessing bug.
Accuracy 0.6630 dropped more than 0.03 below the training baseline 0.8090 ...
prepare_features altered column 'annual_income' (scaling / unit / swap bug?)
  [0]: 76.30683 (ACTUAL), 76306.83 (DESIRED)
Feature 'annual_income' distribution shifted by 2.08 training-stds between training and serving
```

Evidence logs: `reports/bug_demo/*_1_buggy_run.log` (failing), `reports/bug_demo/*_2_fixed_run.log` (passing),
`reports/bug_demo/demo_console_output.txt`, `reports/test_runs/full_suite_passed.log`. These are plain-text
captures of real runs; screenshots of the same terminal output can be taken for submission.

## CI/CD (`.github/workflows/tests.yml`)

On every push/PR, for Python 3.11 and 3.12: checkout → set up Python → install dependencies → train model →
data tests → model tests → API tests → full suite with coverage (fails under 80 %) → upload HTML coverage →
run the bug demo (fails if the framework does *not* catch the bug) → upload demo logs. Any failing step fails the
pipeline.

## Notes / design decisions
* One preprocessing code path serves training and inference; that is what makes train/serve skew testable.
* The API uses strict pydantic types (a string `"35"` is not silently coerced) and a custom 422 handler. The
  latter was added after the tests found that FastAPI's default handler crashed with a server error when the body
  contained `NaN`.
* `models/*.joblib` and generated coverage HTML are git-ignored (CI rebuilds them).
