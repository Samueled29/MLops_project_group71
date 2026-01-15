from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from fruit_and_vegetable_disease.model import Model

app = FastAPI()

MODEL_PATH = Path("models/model.pth")

DEVICE = torch.device("cpu")

# Global model handle (loaded once at startup)
model: Model | None = None


def preprocess_image(file_obj) -> torch.Tensor:
    """
    Preprocess an uploaded image to match the training pipeline.

    Training pipeline summary:
    - Convert to grayscale
    - Resize to 32x32
    - Convert to tensor in [0, 1]
    - Normalize with per-sample mean/std (same as your data.py normalize() usage)
    - Resize to 224x224 and expand channels to RGB (ViT expects 3x224x224)
    """
    # 1) Load image and enforce grayscale
    img = Image.open(file_obj).convert("L")

    # 2) Match the dataset preprocessing (32x32 grayscale)
    img = img.resize((32, 32), Image.BILINEAR)

    # 3) Convert to torch tensor: (1, 1, 32, 32)
    arr = np.array(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    # 4) Normalize exactly like your training preprocessing:
    #    normalize(images) = (images - images.mean()) / images.std()
    # Note: add a small epsilon to avoid division by zero for pathological inputs.
    eps = 1e-8
    x = (x - x.mean()) / (x.std() + eps)

    # 5) Resize to 224x224 and expand channels to RGB
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)

    return x


@app.on_event("startup")
def load_model() -> None:
    """
    Load the model once at application startup.
    This avoids re-loading weights on every request and improves latency.
    """
    global model
    model = Model(num_classes=2).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()


@app.get("/health")
def health() -> dict:
    """
    Liveness probe: returns ok if the process is running.
    Useful for Docker/K8s health checks.
    """
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """
    Readiness probe: returns true only if the model has been loaded.
    Useful for Docker/K8s readiness checks.
    """
    return {"model_loaded": model is not None}


@app.post("/predict")
def predict(file: UploadFile = File(...)) -> dict:
    """
    Predict endpoint:
    - Accepts an image as multipart/form-data
    - Returns predicted label + confidence score
    """
    if model is None:
        # This should not happen if startup event ran successfully,
        # but we keep it defensive.
        return {"error": "Model not loaded"}

    x = preprocess_image(file.file).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        confidence = float(probs.max().item())

    # Keep the label mapping explicit and stable
    label_map = {0: "healthy", 1: "rotten"}

    return {
        "prediction": label_map[pred_idx],
        "confidence": confidence,
    }
