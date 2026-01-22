from __future__ import annotations

import csv
from io import BytesIO
from io import BytesIO
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
from PIL import Image, UnidentifiedImageError

from fruit_and_vegetable_disease.model import Model
from fruit_and_vegetable_disease.aggregate_predictions import aggregate_tensors
import pandas as pd
from evidently.legacy.test_suite import TestSuite
from evidently.legacy.tests import (
    TestNumberOfMissingValues,
    TestShareOfDriftedColumns,
)

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
    # 1) Load image and enforce grayscale
    img = Image.open(file_obj).convert("L")

    # 2) Match the dataset preprocessing (32x32 grayscale)
    img = img.resize((32, 32), Image.BILINEAR)

    # 3) Convert to torch tensor: (1, 1, 32, 32)
    arr = np.array(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    # 4) Normalize per-sample
    eps = 1e-8
    x = (x - x.mean()) / (x.std() + eps)

    # 5) Resize to 224x224 and expand channels to RGB
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)

    return x


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

# Serve frontend (se presente)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
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

    # Controllo content-type (best effort)
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid content-type: {file.content_type}")

    # Limite dimensione upload (best effort: leggiamo in memoria una volta)
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        from io import BytesIO

        x = preprocess_image(BytesIO(data)).to(DEVICE)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file")
    except Exception as e:
        logger.exception("Preprocess failed")
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {type(e).__name__}")

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        confidence = float(probs.max().item())

    label_map = {0: "healthy", 1: "rotten"}
    return PredictResponse(prediction=label_map[pred_idx], confidence=confidence)

# Endpoint to save the predictions to a csv file (separate from /predict for didactic purposes)

CSV_FILE = Path("logs/predictions/predictions_log.csv")
TENSOR_DIR = Path("logs/predictions/tensors")

# Make sure paths exist
CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
TENSOR_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_FILE.exists():
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "image_name", "prediction", "confidence"])

def log_prediction(image_name: str, prediction: str, confidence: float, tensor: torch.Tensor):
    timestamp = datetime.now().isoformat()
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, image_name, prediction, confidence])
    
    # Save preprocessed tensor
    tensor_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{image_name}.pt"
    tensor_path = TENSOR_DIR / tensor_filename
    torch.save(tensor, tensor_path)

@app.post("/predict_log", response_model=PredictResponse)
async def predict_log(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> PredictResponse:
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Check it's an image
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid content-type: {file.content_type}")

    # Read file
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        from io import BytesIO

        x = preprocess_image(BytesIO(data)).to(DEVICE)
    except Exception as e:
        logger.exception("Preprocess failed")
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {type(e).__name__}")

    # Inference
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        confidence = float(probs.max().item())

    label_map = {0: "healthy", 1: "rotten"}
    prediction = label_map[pred_idx]

    # Logging in background (move tensor to CPU for storage)
    background_tasks.add_task(log_prediction, file.filename, prediction, confidence, x.cpu())

    return PredictResponse(prediction=prediction, confidence=confidence)


def _ensure_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x


def _to_features_df(images: torch.Tensor, targets: torch.Tensor) -> pd.DataFrame:
    images = _ensure_nchw(images).float()
    flat = images.view(images.size(0), -1)
    df = pd.DataFrame(
        {
            "pixel_mean": flat.mean(dim=1).cpu().numpy(),
            "pixel_std": flat.std(dim=1, unbiased=False).cpu().numpy(),
        }
    )
    return df


@app.post("/drift_check", response_model=DriftCheckResponse)
def drift_check() -> DriftCheckResponse:
    """Aggregate production predictions and run drift detection tests."""
    try:
        # Aggregate latest prediction tensors
        aggregate_tensors()

        project_root = Path(__file__).resolve().parents[2]
        processed_dir = project_root / "data" / "processed"
        production_dir = project_root / "data" / "production"

        # Load training reference
        if not (processed_dir / "train_images.pt").exists():
            return DriftCheckResponse(
                status="error",
                passed=False,
                message="Training data not found",
            )

        ref_images = torch.load(processed_dir / "train_images.pt")
        ref_targets = torch.load(processed_dir / "train_target.pt")

        # Load production data
        if not (production_dir / "production_images.pt").exists():
            return DriftCheckResponse(
                status="no_data",
                passed=True,
                message="No production data yet",
            )

        prod_images = torch.load(production_dir / "production_images.pt")
        prod_targets = torch.load(production_dir / "production_target.pt")

        reference_data = _to_features_df(ref_images, ref_targets)
        production_data = _to_features_df(prod_images, prod_targets)

        # Run tests
        suite = TestSuite(
            tests=[
                TestNumberOfMissingValues(),
                TestShareOfDriftedColumns(),
            ]
        )
        suite.run(reference_data=reference_data, current_data=production_data)

        # Check if all tests passed
        result = suite.as_dict()
        all_passed = result.get("summary", {}).get("all_passed", True)

        return DriftCheckResponse(
            status="completed",
            passed=all_passed,
            message="Drift check completed successfully"
            if all_passed
            else "Drift detected in production data",
        )

    except Exception as e:
        logger.exception("Drift check failed")
        return DriftCheckResponse(
            status="error", passed=False, message=f"Drift check error: {str(e)}"
        )
