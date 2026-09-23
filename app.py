from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import joblib
import numpy as np

app = FastAPI(title="SilageSense Spoilage Risk Model API")

# Allow the React app (on Netlify, or anywhere) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "silage_extra_trees_model.pkl"
model = joblib.load(MODEL_PATH)

# Must match the exact order/names the model was trained on
FEATURE_ORDER = [
    "pH",
    "dm.s",
    "ammonia.s",
    "lactic.ac.s",
    "acetic.ac.s",
    "propionic.ac.s",
    "butyric.ac.s",
    "ethanol.s",
    "dm.loss",
    "density.1",
    "porosity",
]


@app.get("/")
def root():
    return {"status": "ok", "message": "SilageSense spoilage risk model API is running"}


@app.post("/predict")
def predict(payload: Dict[str, float]):
    missing = [f for f in FEATURE_ORDER if f not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

    features = np.array([[payload[f] for f in FEATURE_ORDER]])

    prediction = int(model.predict(features)[0])  # 0 = Good, 1 = Spoilage Risk

    confidence = None
    try:
        proba = model.predict_proba(features)[0]
        confidence = round(float(max(proba)) * 100, 1)
    except Exception:
        pass

    is_risk = prediction == 1

    return {
        "prediction": prediction,
        "verdict": "Spoilage Risk" if is_risk else "Good Quality",
        "risk_level": "High" if is_risk else "Low",
        "confidence": confidence,
    }
