from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path


import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from transformers import ViTImageProcessorFast


from fruit_and_vegetable_disease.model import Model


image_processor = ViTImageProcessorFast.from_pretrained("google/vit-base-patch16-224")

logger = logging.getLogger("uvicorn.error")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pth"))
DEVICE = torch.device(os.getenv("DEVICE", "cpu"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10MB default

# Frontend paths
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"

# Global model handle
model: Model | None = None


class PredictResponse(BaseModel):
    prediction: str
    confidence: float


class DriftCheckResponse(BaseModel):
    status: str
    passed: bool
    message: str


def preprocess_image(file_obj) -> torch.Tensor:
    img = Image.open(file_obj).convert("RGB")
    inputs = image_processor(img, return_tensors="pt")  # includes resize + normalization
    return inputs["pixel_values"]  # shape: (1, 3, 224, 224)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not MODEL_PATH.exists():
        logger.error("Model file not found at %s", MODEL_PATH)
        model = None
        yield
        return

    logger.info("Loading model from %s on device %s", MODEL_PATH, DEVICE)
    try:
        m = Model(num_classes=2).to(DEVICE)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        m.load_state_dict(state_dict)
        m.eval()
        model = m
        logger.info("Model loaded successfully.")
    except Exception:
        logger.exception("Failed to load model.")
        model = None

    yield

    # cleanup (opzionale)
    model = None


app = FastAPI(lifespan=lifespan)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
INDEX_HTML = FRONTEND_DIR / "index.html"

if FRONTEND_DIR.exists():
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def home():
        if not INDEX_HTML.exists():
            raise HTTPException(status_code=404, detail="Frontend index.html not found")
        return FileResponse(INDEX_HTML)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    return {"model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid content-type: {file.content_type}")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        from io import BytesIO

        x = preprocess_image(BytesIO(data)).to(DEVICE)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file")
    except Exception:
        logger.exception("Preprocess failed")
        raise HTTPException(status_code=400, detail="Preprocessing failed")

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        confidence = float(probs.max().item())

    label_map = {0: "healthy", 1: "rotten"}
    return PredictResponse(prediction=label_map[pred_idx], confidence=confidence)
