"""FastAPI prediction service.   Run:  uvicorn src.api:app --reload"""
import logging
from functools import lru_cache
from typing import Annotated, List, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src import config
from src.model import load_or_train_model, predict_proba

logger = logging.getLogger("loan_api")

R = config.NUMERIC_RANGES
# strict=True: "35" (a string) is rejected instead of silently coerced to 35.
# allow_inf_nan=False: NaN / Infinity are rejected.
StrictInt = lambda lo_hi: Annotated[int, Field(strict=True, ge=lo_hi[0], le=lo_hi[1])]  # noqa: E731
StrictFloat = lambda lo_hi: Annotated[  # noqa: E731
    float, Field(strict=True, allow_inf_nan=False, ge=lo_hi[0], le=lo_hi[1])
]


class LoanApplication(BaseModel):
    """One applicant. Bounds come from the shared data contract in config.py."""

    model_config = ConfigDict(extra="forbid")  # unknown fields -> 422

    age: StrictInt(R["age"])
    annual_income: StrictFloat(R["annual_income"])
    credit_score: StrictInt(R["credit_score"])
    loan_amount: StrictFloat(R["loan_amount"])
    employment_years: StrictInt(R["employment_years"])
    num_late_payments: StrictInt(R["num_late_payments"])
    region: Literal["north", "south", "east", "west"]


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    applications: Annotated[List[LoanApplication], Field(min_length=1, max_length=1000)]


class PredictionResponse(BaseModel):
    prediction: int
    approved: bool
    probability: float


app = FastAPI(title="Loan Approval Prediction API", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return 422 with sanitised errors.

    FastAPI's default handler echoes the offending `input` back, which crashes
    on NaN/Infinity (500 instead of 422) and reflects client data. We keep only
    location, message and type.
    """
    detail = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


@lru_cache(maxsize=1)
def get_model():
    """Dependency: load the model once. Tests override this to inject fakes."""
    return load_or_train_model()


def _score(model, apps: List[LoanApplication]) -> List[PredictionResponse]:
    df = pd.DataFrame([a.model_dump() for a in apps])
    try:
        proba = predict_proba(model, df)[:, 1]
    except Exception as exc:  # never leak internals to the client
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    return [
        PredictionResponse(prediction=int(p >= 0.5), approved=bool(p >= 0.5), probability=round(float(p), 6))
        for p in proba
    ]


def model_dependency():
    try:
        return get_model()
    except Exception as exc:
        logger.exception("Model unavailable")
        raise HTTPException(status_code=503, detail="Model unavailable") from exc


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_one(application: LoanApplication, model=Depends(model_dependency)):
    return _score(model, [application])[0]


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(batch: BatchRequest, model=Depends(model_dependency)):
    return _score(model, batch.applications)
