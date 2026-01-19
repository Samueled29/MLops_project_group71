from fastapi.testclient import TestClient
from fastapi import status

from fruit_and_vegetable_disease import api


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}


def test_ready_endpoint_reflects_model_state(client: TestClient) -> None:
    api.model = object()

    response = client.get("/ready")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"model_loaded": True}

    api.model = None


def test_predict_returns_response(client: TestClient, sample_image_bytes: bytes) -> None:
    class DummyModel:
        def __call__(self, x):
            return api.torch.tensor([[2.0, 1.0]])

    api.model = DummyModel()

    files = {"file": ("sample.png", sample_image_bytes, "image/png")}
    response = client.post("/predict", files=files)

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["prediction"] in {"healthy", "rotten"}
    assert 0.0 <= body["confidence"] <= 1.0

    api.model = None


def test_predict_rejects_non_image_content_type(client: TestClient) -> None:
    api.model = object()

    files = {"file": ("text.txt", b"hello", "text/plain")}
    response = client.post("/predict", files=files)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid content-type" in response.json()["detail"]

    api.model = None


def test_predict_returns_503_when_model_missing(client: TestClient, sample_image_bytes: bytes) -> None:
    api.model = None

    files = {"file": ("sample.png", sample_image_bytes, "image/png")}
    response = client.post("/predict", files=files)

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"] == "Model not loaded"
