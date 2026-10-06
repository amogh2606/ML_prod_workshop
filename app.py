import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path(__file__).resolve().parent / "iris_model.pkl"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

app = FastAPI(title="Iris Prediction API")


with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root() -> dict:
    return {"message": "Iris model is ready. POST to /predict with 4 flower measurements."}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(features: IrisFeatures) -> dict:
    sample = np.array(
        [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]],
        dtype=float,
    )

    prediction_idx = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]

    result = {
        "prediction": CLASS_NAMES[prediction_idx],
        "probabilities": {
            label: float(prob)
            for label, prob in zip(CLASS_NAMES, probabilities)
        },
    }
    return result
