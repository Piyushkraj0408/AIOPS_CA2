from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

app = FastAPI(
    title="Breast Cancer Classification API",
    description="Educational demo only. Not for medical diagnosis or treatment decisions.",
    version="1.0.0",
)

# The saved artifact is a dict containing the fitted pipeline and feature metadata.
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
FEATURE_NAMES = artifact["feature_names"]
CLASS_NAMES = artifact["class_names"]


class PredictionRequest(BaseModel):
    # Exactly 30 numeric features are required, using the feature names from the model.
    features: dict[str, Annotated[float, Field(allow_inf_nan=False)]]


@app.get("/")
def root():
    return {"message": "Breast Cancer Classification API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictionRequest):
    supplied = set(request.features)
    expected = set(FEATURE_NAMES)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"message": "Input features do not match model schema",
                    "missing_features": missing, "extra_features": extra},
        )

    row = pd.DataFrame([[request.features[name] for name in FEATURE_NAMES]],
                       columns=FEATURE_NAMES)
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    return {
        "prediction": CLASS_NAMES[prediction],
        "prediction_id": prediction,
        "probabilities": {
            CLASS_NAMES[i]: round(float(probabilities[i]), 6)
            for i in range(len(CLASS_NAMES))
        },
        "disclaimer": "Educational demo only; not a medical diagnostic tool.",
    }
