import io
import sys
from pathlib import Path
from locust import HttpUser, task, between
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def create_test_image() -> bytes:
    """Create a small grayscale PNG image for testing."""
    img = Image.new("L", (32, 32), color=128)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class APIUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def health_check(self) -> None:
        """Health check endpoint - high frequency."""
        self.client.get("/health")

    @task(2)
    def readiness_check(self) -> None:
        """Readiness check endpoint - medium frequency."""
        self.client.get("/ready")

    @task(5)
    def predict_image(self) -> None:
        """Prediction endpoint - high frequency."""
        image_bytes = create_test_image()
        files = {"file": ("test.png", image_bytes, "image/png")}
        self.client.post("/predict", files=files)
