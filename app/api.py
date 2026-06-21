"""
Optional FastAPI endpoint, kept alongside the Streamlit app.

The team's working prototype is the Streamlit app (app.py) — it's what gets
demoed live with no GPU dependency. This FastAPI wrapper exposes the same
DRPredictor as a REST endpoint, in case the judges want to see an API-style
integration point (e.g. for a future React frontend) on top of the same model.

Run with:  uvicorn api:app --reload --port 8000
Then POST an image to http://localhost:8000/predict
"""

from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import DRPredictor

MODEL_PATH = Path(__file__).parent / "model" / "dr_model_traced.pt"

app = FastAPI(
    title="DR Detection API",
    description="Biothon 2026 — Problem P2: Early-Stage Diabetic Retinopathy Detection",
    version="1.0.0",
)

# Permissive CORS for local hackathon demo purposes only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor = None


def get_predictor() -> DRPredictor:
    global _predictor
    if _predictor is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model not found at {MODEL_PATH}. Train it in the Colab notebook, "
                    "download dr_model_traced.pt, and place it at app/model/dr_model_traced.pt."
                ),
            )
        _predictor = DRPredictor(model_path=str(MODEL_PATH), img_size=224, device="cpu")
    return _predictor


class PredictionResponse(BaseModel):
    grade: int
    grade_label: str
    raw_score: float
    confidence: float
    probability_bars: dict
    needs_manual_review: bool
    clinical_recommendation: str
    dr_present: bool


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Upload a PNG or JPEG image.")

    contents = await file.read()
    file_array = np.frombuffer(contents, dtype=np.uint8)
    bgr_img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if bgr_img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    predictor = get_predictor()
    result = predictor.predict(bgr_img)

    return PredictionResponse(
        grade=result.grade,
        grade_label=result.grade_label,
        raw_score=result.raw_score,
        confidence=result.confidence,
        probability_bars=result.probability_bars,
        needs_manual_review=result.needs_manual_review,
        clinical_recommendation=result.clinical_recommendation,
        dr_present=result.dr_present,
    )
