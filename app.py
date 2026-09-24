from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import pickle
import numpy as np

app = FastAPI(title="SilageSense Spoilage Risk Model API")

# Allow the frontend (Netlify, or anywhere) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "silage_sensor_model.pkl"

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
FEATURE_ORDER = bundle["feature_cols"]  # ["N", "P", "K", "Moisture", "Temperature", "pH", "EC"]


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "SilageSense spoilage risk model API is running",
        "model_name": bundle.get("model_name"),
        "features_expected": FEATURE_ORDER,
    }


@app.post("/predict")
def predict(payload: Dict[str, float]):
    missing = [f for f in FEATURE_ORDER if f not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

    features = np.array([[payload[f] for f in FEATURE_ORDER]])

    prediction = int(model.predict(features)[0])  # 0 = Good, 1 = Quality Risk

    confidence = None
    try:
        proba = model.predict_proba(features)[0]
        confidence = round(float(max(proba)) * 100, 1)
    except Exception:
        pass

    is_risk = prediction == 1

    return {
        "prediction": prediction,
        "verdict": "Quality Risk" if is_risk else "Good Quality",
        "risk_level": "High" if is_risk else "Low",
        "confidence": confidence,
    }
