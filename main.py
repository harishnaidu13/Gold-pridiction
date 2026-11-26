from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

import custom_transformers  # noqa: F401  (needed so joblib can unpickle the pipeline)

MODEL_PATH = Path(__file__).parent / "model.joblib"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError("model.joblib not found. Run `py train.py` first.")
    state.update(joblib.load(MODEL_PATH))
    yield
    state.clear()


app = FastAPI(title="Gold Close Prediction API", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    date: Optional[str] = Field(
        None,
        description="Optional date (day-first, e.g. 17-11-2026). Fills day/month/year.",
    )
    features: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Feature name -> value. Missing features are imputed by the model.",
    )


class PredictResponse(BaseModel):
    gold_close_prediction: float


@app.get("/")
def root():
    return {"message": "Gold Close Prediction API. Go to /docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    """List the input columns the model expects, plus its test metrics."""
    return {"features": state["features"], "metrics": state["metrics"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    expected = state["features"]

    unknown = set(req.features) - set(expected)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown features: {sorted(unknown)}. See GET /features.",
        )

    row = {name: np.nan for name in expected}

    if req.date:
        parsed = pd.to_datetime(req.date, errors="coerce", dayfirst=True)
        if pd.isna(parsed):
            raise HTTPException(status_code=422, detail="Could not parse date.")
        for key, value in (("day", parsed.day), ("month", parsed.month), ("year", parsed.year)):
            if key in row:
                row[key] = value

    for name, value in req.features.items():
        if value is not None:
            row[name] = value

    X = pd.DataFrame([row], columns=expected)
    prediction = state["pipeline"].predict(X)[0]
    return PredictResponse(gold_close_prediction=float(prediction))


# Show every real feature name in the /docs example body
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    if "features" in state:
        schema["components"]["schemas"]["PredictRequest"]["example"] = {
            "date": "17-11-2026",
            "features": {name: None for name in state["features"]},
        }
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi