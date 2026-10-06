import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "iris_model.pkl"
STATIC_DIR = BASE_DIR / "static"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

app = FastAPI(title="Iris Prediction API")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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
