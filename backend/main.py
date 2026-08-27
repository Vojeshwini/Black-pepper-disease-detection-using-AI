r"""
PepperAI Backend API (FastAPI)
Serves three things over local HTTP endpoints:
  1. Disease detection  -> /predict-disease   (uses best_model.pt, EfficientNet-B0 CNN)
  2. Yield prediction    -> /predict-yield     (uses yield_model.json, XGBoost)
  3. Recommendations     -> /recommend         (pure Python rules engine)
Also serves the website itself at "/".

Run with:
    conda activate pepperai
    cd /d D:\PepperAI\backend
    uvicorn main:app --reload
Then open http://localhost:8000 in your browser.
"""

import os
import io
import json

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

import xgboost as xgb
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from recommendation_engine import get_recommendation

# ---------------------------------------------------------------------------
# PATHS -- adjust these if your files are saved elsewhere
# ---------------------------------------------------------------------------
BASE_DIR = r"D:\PepperAI"
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")
YIELD_MODEL_PATH = os.path.join(BASE_DIR, "yield_model.json")
DISTRICT_ENCODING_PATH = os.path.join(BASE_DIR, "district_encoding.json")
WEBSITE_DIR = os.path.join(BASE_DIR, "website")   # where pepperai-website.html lives

CLASS_NAMES = ["foot_rot", "healthy", "leaf_blight", "pollu_disease", "slow_decline", "yellow_mottle_virus"]
IMG_SIZE = 224

# ---------------------------------------------------------------------------
# APP SETUP
# ---------------------------------------------------------------------------
app = FastAPI(title="PepperAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# LOAD DISEASE DETECTION MODEL (CNN) ONCE AT STARTUP
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

disease_model = efficientnet_b0(weights=None)
disease_model.classifier[1] = nn.Linear(disease_model.classifier[1].in_features, len(CLASS_NAMES))

if os.path.exists(DISEASE_MODEL_PATH):
    disease_model.load_state_dict(torch.load(DISEASE_MODEL_PATH, map_location=device))
    disease_model.to(device)
    disease_model.eval()
    print(f"[startup] Disease model loaded from {DISEASE_MODEL_PATH} on {device}")
else:
    print(f"[startup] WARNING: disease model not found at {DISEASE_MODEL_PATH}")

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# LOAD YIELD PREDICTION MODEL (XGBoost) ONCE AT STARTUP
# ---------------------------------------------------------------------------
yield_model = xgb.XGBRegressor()
if os.path.exists(YIELD_MODEL_PATH):
    yield_model.load_model(YIELD_MODEL_PATH)
    print(f"[startup] Yield model loaded from {YIELD_MODEL_PATH}")
else:
    print(f"[startup] WARNING: yield model not found at {YIELD_MODEL_PATH}")

district_map, state_map = {}, {}
district_map_lower, state_map_lower = {}, {}
if os.path.exists(DISTRICT_ENCODING_PATH):
    with open(DISTRICT_ENCODING_PATH) as f:
        enc = json.load(f)
        district_map = {v: int(k) for k, v in enc["district_map"].items()}
        state_map = {v: int(k) for k, v in enc["state_map"].items()}
        district_map_lower = {k.lower(): v for k, v in district_map.items()}
        state_map_lower = {k.lower(): v for k, v in state_map.items()}
    print(f"[startup] Loaded {len(district_map)} districts, {len(state_map)} states")
else:
    print(f"[startup] WARNING: district encoding not found at {DISTRICT_ENCODING_PATH}")

# ---------------------------------------------------------------------------
# ENDPOINT 1 -- DISEASE DETECTION
# ---------------------------------------------------------------------------
@app.post("/predict-disease")
async def predict_disease(file: UploadFile = File(...)):
    if not os.path.exists(DISEASE_MODEL_PATH):
        raise HTTPException(status_code=503, detail="Disease model not loaded on server.")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded image.")

    input_tensor = eval_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = disease_model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

    all_class_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": round(confidence, 4),
        "all_class_probabilities": {k: round(v, 4) for k, v in all_class_probs.items()},
    }


# ---------------------------------------------------------------------------
# ENDPOINT 2 -- YIELD PREDICTION
# ---------------------------------------------------------------------------
class YieldInput(BaseModel):
    district: str
    state: str
    area_ha: float
    rainfall_mm: float
    temp_c: float
    prev_yield_t_ha: float
    year: int = 2026


@app.post("/predict-yield")
async def predict_yield(payload: YieldInput):
    if not os.path.exists(YIELD_MODEL_PATH):
        raise HTTPException(status_code=503, detail="Yield model not loaded on server.")

    district_code = district_map_lower.get(payload.district.strip().lower(), -1)
    state_code = state_map_lower.get(payload.state.strip().lower(), -1)

    features = np.array([[
        payload.area_ha,
        payload.rainfall_mm,
        payload.temp_c,
        payload.prev_yield_t_ha,
        district_code,
        state_code,
        payload.year,
    ]])

    predicted_yield = float(yield_model.predict(features)[0])
    predicted_total = predicted_yield * payload.area_ha

    return {
        "predicted_yield_t_ha": round(predicted_yield, 3),
        "predicted_total_tonnes": round(predicted_total, 2),
        "district_recognized": district_code != -1,
        "state_recognized": state_code != -1,
    }


# ---------------------------------------------------------------------------
# ENDPOINT 3 -- RECOMMENDATIONS
# ---------------------------------------------------------------------------
class RecommendationInput(BaseModel):
    disease: str
    soil_type: str
    rainfall_mm_per_month: float
    vine_age_years: int = 7
    severity: str = "medium"


@app.post("/recommend")
async def recommend(payload: RecommendationInput):
    try:
        result = get_recommendation(
            disease=payload.disease,
            soil_type=payload.soil_type,
            rainfall_mm_per_month=payload.rainfall_mm_per_month,
            vine_age_years=payload.vine_age_years,
            severity=payload.severity,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "disease_model_loaded": os.path.exists(DISEASE_MODEL_PATH),
        "yield_model_loaded": os.path.exists(YIELD_MODEL_PATH),
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# SERVE THE WEBSITE ITSELF
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_website():
    index_path = os.path.join(WEBSITE_DIR, "pepperai-website.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "PepperAI backend is running. Website file not found at " + index_path}
