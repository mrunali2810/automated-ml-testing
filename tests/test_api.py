"""API tests (FastAPI TestClient): valid requests, every kind of invalid
request, status codes and response structure."""
import pytest

from src.api import app, get_model, model_dependency

pytestmark = pytest.mark.api

FIELDS = ["age", "annual_income", "credit_score", "loan_amount",
          "employment_years", "num_late_payments", "region"]


# ---------------------------------------------------------------- happy path
def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_valid_prediction_request(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"prediction", "approved", "probability"}, f"Unexpected response keys: {set(body)}"
    assert body["prediction"] in (0, 1)
    assert isinstance(body["approved"], bool)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["approved"] == (body["prediction"] == 1) == (body["probability"] >= 0.5)


def test_good_applicant_approved_and_bad_rejected(client, payload):
    good = client.post("/predict", json={**payload, "credit_score": 820, "annual_income": 150000.0,
                                         "loan_amount": 5000.0, "employment_years": 20}).json()
    bad = client.post("/predict", json={**payload, "credit_score": 400, "annual_income": 20000.0,
                                        "loan_amount": 100000.0, "num_late_payments": 12}).json()
    assert good["approved"] is True and bad["approved"] is False


def test_integer_accepted_for_float_field(client, payload):
    payload["annual_income"] = 60000  # JSON int for a float field is fine
    assert client.post("/predict", json=payload).status_code == 200


def test_boundary_values_accepted(client, payload):
    payload.update(age=18, credit_score=300, annual_income=10000.0, loan_amount=500.0,
                   employment_years=0, num_late_payments=0)
    assert client.post("/predict", json=payload).status_code == 200


def test_batch_prediction(client, payload):
    r = client.post("/predict/batch", json={"applications": [payload, {**payload, "credit_score": 450}]})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and len(body) == 2
    assert body[0]["probability"] > body[1]["probability"], "Lower credit score should lower probability"


def test_batch_empty_list_rejected(client):
    assert client.post("/predict/batch", json={"applications": []}).status_code == 422


# ------------------------------------------------------------ missing fields
@pytest.mark.parametrize("field", FIELDS)
def test_missing_required_field_returns_422(client, payload, field):
    del payload[field]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422, f"Missing '{field}' should be rejected, got {r.status_code}"
    assert any(field in str(err["loc"]) for err in r.json()["detail"])


# ----------------------------------------------------------- invalid types
@pytest.mark.parametrize("field, bad", [
    ("age", "thirty"), ("age", "35"), ("age", 35.5), ("age", True), ("age", [35]),
    ("annual_income", "lots"), ("annual_income", {"v": 1}),
    ("credit_score", "high"), ("loan_amount", "a lot"),
    ("employment_years", "many"), ("num_late_payments", None),
    ("region", 5), ("region", ["north"]),
])
def test_invalid_data_types_return_422(client, payload, field, bad):
    payload[field] = bad
    r = client.post("/predict", json=payload)
    assert r.status_code == 422, f"{field}={bad!r} should be rejected, got {r.status_code}: {r.text}"


# ------------------------------------------------------------------- nulls
@pytest.mark.parametrize("field", FIELDS)
def test_null_values_return_422(client, payload, field):
    payload[field] = None
    assert client.post("/predict", json=payload).status_code == 422


# ----------------------------------------------------------- out of range
@pytest.mark.parametrize("field, bad", [
    ("age", 17), ("age", 120), ("age", -5),
    ("annual_income", -1.0), ("annual_income", 5.0), ("annual_income", 10_000_000.0),
    ("credit_score", 299), ("credit_score", 900),
    ("loan_amount", 0.0), ("loan_amount", 1e9),
    ("employment_years", -1), ("employment_years", 80),
    ("num_late_payments", -1), ("num_late_payments", 99),
])
def test_out_of_range_values_return_422(client, payload, field, bad):
    payload[field] = bad
    r = client.post("/predict", json=payload)
    assert r.status_code == 422, f"{field}={bad} should be rejected, got {r.status_code}"


def test_invalid_region_returns_422(client, payload):
    payload["region"] = "atlantis"
    assert client.post("/predict", json=payload).status_code == 422


def test_nan_and_infinity_rejected(client, payload):
    for token in ("NaN", "Infinity", "-Infinity"):
        raw = '{"age":35,"annual_income":%s,"credit_score":700,"loan_amount":15000.0,' \
              '"employment_years":8,"num_late_payments":0,"region":"north"}' % token
        r = client.post("/predict", content=raw, headers={"Content-Type": "application/json"})
        assert r.status_code == 422, f"{token} should be rejected, got {r.status_code}"


def test_unknown_extra_field_rejected(client, payload):
    payload["is_vip"] = True
    assert client.post("/predict", json=payload).status_code == 422


# ----------------------------------------------------- empty / invalid body
def test_empty_json_object_returns_422(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422
    assert len(r.json()["detail"]) == len(FIELDS), "Every missing field should be reported"


def test_no_body_returns_422(client):
    assert client.post("/predict").status_code == 422


@pytest.mark.parametrize("body", [b"{not valid json", b"", b"[1, 2, 3]", b'"just a string"', b"null"])
def test_invalid_json_returns_422(client, body):
    r = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422, f"body={body!r} -> {r.status_code}"


# --------------------------------------------------------- HTTP semantics
def test_get_on_predict_returns_405(client):
    assert client.get("/predict").status_code == 405


def test_unknown_route_returns_404(client):
    assert client.get("/nope").status_code == 404


def test_validation_error_has_structured_detail(client):
    body = client.post("/predict", json={}).json()
    assert "detail" in body and all({"loc", "msg", "type"} <= set(e) for e in body["detail"])


# ---------------------------------------------------- server-side failures
class _BrokenModel:
    def predict_proba(self, X):
        raise RuntimeError("boom - internal detail that must not leak")

    def predict(self, X):
        raise RuntimeError("boom")


def test_model_failure_returns_500_without_leaking_internals(client, payload):
    app.dependency_overrides[model_dependency] = lambda: _BrokenModel()
    try:
        r = client.post("/predict", json=payload)
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 500
    assert "boom" not in r.text, "Internal error details leaked to the client"


def test_unavailable_model_returns_503(client, payload, monkeypatch):
    def fail():
        raise FileNotFoundError("model missing")

    get_model.cache_clear()
    monkeypatch.setattr("src.api.load_or_train_model", fail)
    try:
        r = client.post("/predict", json=payload)
    finally:
        get_model.cache_clear()
    assert r.status_code == 503
