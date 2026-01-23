import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from google.cloud import storage
import pandas as pd
import torch
from pydantic import BaseModel

from evidently.legacy.test_suite import TestSuite
from evidently.legacy.tests import TestNumberOfMissingValues, TestShareOfDriftedColumns

logger = logging.getLogger("uvicorn.error")

BUCKET_NAME = "fruit-and-veg-disease_bucket_predictions"
PREDICTIONS_PREFIX = "predictions"

app = FastAPI()


class PredictionLog(BaseModel):
    image_tensor: list  # Flattened tensor as list
    prediction: int
    timestamp: str = None


class DriftCheckResponse(BaseModel):
    status: str
    passed: bool
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_log")
def predict_log(data: PredictionLog):
    """Save a prediction to GCP bucket."""
    try:
        if data.timestamp is None:
            data.timestamp = datetime.now().isoformat()

        # Save to bucket
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        # Create unique filename
        filename = f"{PREDICTIONS_PREFIX}/prediction_{data.timestamp.replace(':', '-').replace('.', '_')}.pt"

        # Convert back to tensor and save to temp file
        tensor = torch.tensor(data.image_tensor)
        temp_path = Path(f"/tmp/pred_{data.timestamp.replace(':', '-').replace('.', '_')}.pt")
        
        torch.save(
            {"image": tensor, "prediction": data.prediction, "timestamp": data.timestamp}, 
            temp_path
        )

        # Upload to bucket
        blob = bucket.blob(filename)
        blob.upload_from_filename(temp_path)
        temp_path.unlink()  # Clean up temp file

        logger.info(f"Saved prediction to {filename}")
        return {"status": "saved", "filename": filename}

    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")
        return {"status": "error", "message": str(e)}


def _to_features_df(images: torch.Tensor, targets: torch.Tensor) -> pd.DataFrame:
    """Convert images to feature dataframe for drift detection."""
    # Ensure NCHW format (batch, channels, height, width)
    if images.dim() == 3:
        images = images.unsqueeze(1)

    images = images.float()
    flat = images.view(images.size(0), -1)

    df = pd.DataFrame(
        {
            "pixel_mean": flat.mean(dim=1).cpu().numpy(),
            "pixel_std": flat.std(dim=1, unbiased=False).cpu().numpy(),
        }
    )
    return df


@app.post("/drift_check", response_model=DriftCheckResponse)
def drift_check(n_predictions: int = 100):
    """Check for drift between training data and recent predictions from bucket."""
    try:
        # Load training data
        project_root = Path(__file__).resolve().parents[2]
        processed_dir = project_root / "data" / "processed"
        
        train_images = torch.load(processed_dir / "train_images.pt")
        train_targets = torch.load(processed_dir / "train_target.pt")

        # Download recent predictions from bucket
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix=PREDICTIONS_PREFIX))

        if not blobs:
            return DriftCheckResponse(
                status="no_data", passed=True, message="No predictions found in bucket"
            )

        # Sort by time (most recent first) and take last N
        blobs.sort(key=lambda x: x.updated, reverse=True)
        recent_blobs = blobs[:n_predictions]

        # Download and aggregate predictions
        pred_images = []
        pred_targets = []

        for blob in recent_blobs:
            temp_path = Path(f"/tmp/{blob.name.split('/')[-1]}")
            blob.download_to_filename(temp_path)
            data = torch.load(temp_path)
            pred_images.append(data["image"])
            pred_targets.append(data["prediction"])
            temp_path.unlink()

        if not pred_images:
            return DriftCheckResponse(
                status="no_data", passed=True, message="No valid predictions found"
            )

        # Stack into tensors
        prod_images = torch.stack(pred_images)
        prod_targets = torch.tensor(pred_targets)

        # Convert to features
        reference_data = _to_features_df(train_images, train_targets)
        production_data = _to_features_df(prod_images, prod_targets)

        # Run drift tests
        suite = TestSuite(
            tests=[
                TestNumberOfMissingValues(),
                TestShareOfDriftedColumns(),
            ]
        )
        suite.run(reference_data=reference_data, current_data=production_data)

        # Check results
        result = suite.as_dict()
        all_passed = result.get("summary", {}).get("all_passed", True)

        return DriftCheckResponse(
            status="completed",
            passed=all_passed,
            message=f"Drift check completed on {len(pred_images)} predictions"
            if all_passed
            else f"Drift detected in {len(pred_images)} predictions",
        )

    except Exception as e:
        logger.exception("Drift check failed")
        return DriftCheckResponse(status="error", passed=False, message=f"Error: {str(e)}")
