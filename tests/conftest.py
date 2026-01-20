import sys
from pathlib import Path
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def client():
    from fruit_and_vegetable_disease.api import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def sample_image_bytes() -> bytes:
    """Create a small grayscale PNG image and return its bytes."""
    img = Image.new("L", (32, 32), color=128)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()
