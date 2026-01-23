import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def drift_client():
    """Create a test client for the drift API."""
    from fruit_and_vegetable_disease.drift_api import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def sample_tensor():
    """Create a sample image tensor."""
    return torch.randn(1, 3, 32, 32)


@pytest.fixture()
def sample_target():
    """Create a sample target tensor."""
    return torch.tensor([0])


class TestToFeaturesDf:
    """Tests for the _to_features_df function."""

    def test_to_features_df_with_4d_tensor(self, sample_tensor, sample_target):
        """Test conversion of 4D image tensor to features dataframe."""
        from fruit_and_vegetable_disease.drift_api import _to_features_df

        df = _to_features_df(sample_tensor, sample_target)

        assert isinstance(df, pd.DataFrame)
        assert "pixel_mean" in df.columns
        assert "pixel_std" in df.columns
        assert len(df) == 1

    def test_to_features_df_with_3d_tensor(self, sample_target):
        """Test conversion of 3D image tensor to features dataframe."""
        from fruit_and_vegetable_disease.drift_api import _to_features_df

        tensor_3d = torch.randn(1, 32, 32)
        df = _to_features_df(tensor_3d, sample_target)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df["pixel_mean"].iloc[0].is_integer() is False

    def test_to_features_df_multiple_samples(self, sample_target):
        """Test conversion of multiple samples."""
        from fruit_and_vegetable_disease.drift_api import _to_features_df

        tensor = torch.randn(5, 3, 32, 32)
        target = torch.arange(5)
        df = _to_features_df(tensor, target)

        assert len(df) == 5
        assert all(col in df.columns for col in ["pixel_mean", "pixel_std"])

    def test_to_features_df_output_shapes(self, sample_tensor, sample_target):
        """Test that output has correct shape and types."""
        from fruit_and_vegetable_disease.drift_api import _to_features_df

        df = _to_features_df(sample_tensor, sample_target)

        assert df.shape[0] == sample_tensor.shape[0]
        assert df.shape[1] == 2
        assert df["pixel_mean"].dtype in [float, "float64", "float32"]


class TestPredictionLogModel:
    """Tests for the PredictionLog Pydantic model."""

    def test_prediction_log_with_timestamp(self):
        """Test PredictionLog creation with timestamp."""
        from fruit_and_vegetable_disease.drift_api import PredictionLog

        log = PredictionLog(
            image_tensor=[1.0, 2.0, 3.0],
            prediction=1,
            timestamp="2024-01-20T10:30:00",
        )

        assert log.image_tensor == [1.0, 2.0, 3.0]
        assert log.prediction == 1
        assert log.timestamp == "2024-01-20T10:30:00"

    def test_prediction_log_without_timestamp(self):
        """Test PredictionLog creation without timestamp."""
        from fruit_and_vegetable_disease.drift_api import PredictionLog

        log = PredictionLog(image_tensor=[1.0, 2.0], prediction=0)

        assert log.image_tensor == [1.0, 2.0]
        assert log.prediction == 0
        assert log.timestamp is None

    def test_prediction_log_validation(self):
        """Test PredictionLog validation."""
        from fruit_and_vegetable_disease.drift_api import PredictionLog

        log = PredictionLog(image_tensor=[0.5] * 1024, prediction=2)
        assert len(log.image_tensor) == 1024


class TestPredictLogEndpoint:
    """Tests for the /predict_log endpoint."""

    @patch("fruit_and_vegetable_disease.drift_api.storage.Client")
    @patch("torch.save")
    @patch("pathlib.Path.unlink")
    def test_predict_log_saves_successfully(self, mock_unlink, mock_torch_save, mock_storage_client, drift_client):
        """Test that predict_log saves prediction to bucket."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client

        payload = {
            "image_tensor": [1.0, 2.0, 3.0] * 342,  # ~1024 elements
            "prediction": 1,
        }

        response = drift_client.post("/predict_log", json=payload)

        assert response.status_code == 200
        assert response.json()["status"] == "saved"
        assert "predictions/prediction_" in response.json()["filename"]

    @patch("fruit_and_vegetable_disease.drift_api.storage.Client")
    def test_predict_log_handles_error(self, mock_storage_client, drift_client):
        """Test that predict_log handles errors gracefully."""
        mock_client = MagicMock()
        mock_client.bucket.side_effect = Exception("Connection error")
        mock_storage_client.return_value = mock_client

        payload = {"image_tensor": [1.0, 2.0], "prediction": 0}

        response = drift_client.post("/predict_log", json=payload)

        assert response.status_code == 200
        assert response.json()["status"] == "error"
        assert "message" in response.json()

    @patch("fruit_and_vegetable_disease.drift_api.storage.Client")
    @patch("torch.save")
    @patch("pathlib.Path.unlink")
    def test_predict_log_uses_provided_timestamp(self, mock_unlink, mock_torch_save, mock_storage_client, drift_client):
        """Test that predict_log uses provided timestamp."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client

        timestamp = "2024-01-20T10:30:00.123456"
        payload = {
            "image_tensor": [1.0, 2.0] * 512,
            "prediction": 1,
            "timestamp": timestamp,
        }

        response = drift_client.post("/predict_log", json=payload)

        assert response.status_code == 200
        saved_data = mock_torch_save.call_args[0][0]
        assert saved_data["timestamp"] == timestamp

    @patch("fruit_and_vegetable_disease.drift_api.storage.Client")
    @patch("torch.save")
    @patch("pathlib.Path.unlink")
    def test_predict_log_generates_timestamp_if_none(
        self, mock_unlink, mock_torch_save, mock_storage_client, drift_client
    ):
        """Test that predict_log generates timestamp if none provided."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client

        payload = {"image_tensor": [1.0, 2.0] * 512, "prediction": 1}

        response = drift_client.post("/predict_log", json=payload)

        assert response.status_code == 200
        saved_data = mock_torch_save.call_args[0][0]
        assert "timestamp" in saved_data
        assert saved_data["timestamp"] is not None


class TestDriftCheckEndpoint:
    """Tests for the /drift_check endpoint."""

    @patch("fruit_and_vegetable_disease.drift_api.TestSuite")
    @patch("fruit_and_vegetable_disease.drift_api.storage.Client")
    @patch("torch.load")
    @patch("torch.stack")
    def test_drift_check_no_predictions(
        self, mock_stack, mock_torch_load, mock_storage_client, mock_test_suite, drift_client
    ):
        """Test drift check when no predictions are found."""
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = []
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client

        response = drift_client.post("/drift_check", json={"n_predictions": 100})

        assert response.status_code == 200
        assert response.json()["status"] == "no_data"
        assert response.json()["passed"] is True

    @patch("fruit_and_vegetable_disease.drift_api.TestSuite")
    @patch("fruit_and_vegetable_disease.drift_api.storage.Client")
    @patch("torch.load")
    @patch("torch.stack")
    @patch("pathlib.Path.unlink")
    def test_drift_check_successful(
        self,
        mock_unlink,
        mock_stack,
        mock_torch_load,
        mock_storage_client,
        mock_test_suite,
        drift_client,
        sample_tensor,
        sample_target,
    ):
        """Test successful drift check."""
        # Setup GCS mock
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.name = "predictions/prediction_1"
        mock_blob.updated = datetime.now()
        mock_bucket.list_blobs.return_value = [mock_blob]
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client

        # Setup torch.load mock
        mock_torch_load.return_value = {"image": sample_tensor, "prediction": 0}
        mock_stack.return_value = sample_tensor

        # Setup TestSuite mock
        mock_suite_instance = MagicMock()
        mock_suite_instance.as_dict.return_value = {"summary": {"all_passed": True}}
        mock_test_suite.return_value = mock_suite_instance

        with patch("torch.load") as torch_load_patch:
            torch_load_patch.side_effect = [
                sample_tensor,  # train_images
                sample_target,  # train_targets
                {"image": sample_tensor, "prediction": 0},  # prediction data
            ]

            response = drift_client.post("/drift_check", json={"n_predictions": 1})

            assert response.status_code == 200
            assert response.json()["status"] == "completed"
            assert response.json()["passed"] is True

    @patch("fruit_and_vegetable_disease.drift_api.TestSuite")
    @patch("fruit_and_vegetable_disease.drift_api.storage.Client")
    @patch("torch.load")
    @patch("torch.stack")
    @patch("pathlib.Path.unlink")
    def test_drift_check_detects_drift(
        self,
        mock_unlink,
        mock_stack,
        mock_torch_load,
        mock_storage_client,
        mock_test_suite,
        drift_client,
        sample_tensor,
        sample_target,
    ):
        """Test drift check when drift is detected."""
        # Setup GCS mock
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.name = "predictions/prediction_1"
        mock_blob.updated = datetime.now()
        mock_bucket.list_blobs.return_value = [mock_blob]
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client

        # Setup torch.load mock
        mock_torch_load.return_value = {"image": sample_tensor, "prediction": 0}
        mock_stack.return_value = sample_tensor

        # Setup TestSuite mock - no drift passed
        mock_suite_instance = MagicMock()
        mock_suite_instance.as_dict.return_value = {"summary": {"all_passed": False}}
        mock_test_suite.return_value = mock_suite_instance

        with patch("torch.load") as torch_load_patch:
            torch_load_patch.side_effect = [
                sample_tensor,  # train_images
                sample_target,  # train_targets
                {"image": sample_tensor, "prediction": 0},  # prediction data
            ]

            response = drift_client.post("/drift_check", json={"n_predictions": 1})

            assert response.status_code == 200
            assert response.json()["status"] == "completed"
            assert response.json()["passed"] is False
            assert "Drift detected" in response.json()["message"]

    @patch("fruit_and_vegetable_disease.drift_api.TestSuite")
    @patch("fruit_and_vegetable_disease.drift_api.storage.Client")
    def test_drift_check_error_handling(self, mock_storage_client, mock_test_suite, drift_client):
        """Test drift check error handling."""
        mock_client = MagicMock()
        mock_client.bucket.side_effect = Exception("GCS connection error")
        mock_storage_client.return_value = mock_client

        response = drift_client.post("/drift_check", json={"n_predictions": 100})

        assert response.status_code == 200
        assert response.json()["status"] == "error"
        assert response.json()["passed"] is False
        assert "Error" in response.json()["message"]

    @patch("fruit_and_vegetable_disease.drift_api.TestSuite")
    @patch("fruit_and_vegetable_disease.drift_api.storage.Client")
    @patch("torch.load")
    @patch("torch.stack")
    @patch("pathlib.Path.unlink")
    def test_drift_check_empty_predictions(
        self,
        mock_unlink,
        mock_stack,
        mock_torch_load,
        mock_storage_client,
        mock_test_suite,
        drift_client,
        sample_tensor,
        sample_target,
    ):
        """Test drift check with no valid predictions after download."""
        # Setup GCS mock with blob
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.name = "predictions/prediction_1"
        mock_blob.updated = datetime.now()
        mock_bucket.list_blobs.return_value = [mock_blob]
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client

        # Mock download to raise error
        mock_blob.download_to_filename.side_effect = Exception("Download failed")

        with patch("torch.load") as torch_load_patch:
            torch_load_patch.side_effect = [
                sample_tensor,  # train_images
                sample_target,  # train_targets
            ]

            response = drift_client.post("/drift_check", json={"n_predictions": 1})

            assert response.status_code == 200
            assert response.json()["status"] == "error"
            assert response.json()["passed"] is False

    @patch("fruit_and_vegetable_disease.drift_api.TestSuite")
    @patch("fruit_and_vegetable_disease.drift_api.storage.Client")
    @patch("torch.load")
    @patch("torch.stack")
    @patch("pathlib.Path.unlink")
    def test_drift_check_default_n_predictions(
        self,
        mock_unlink,
        mock_stack,
        mock_torch_load,
        mock_storage_client,
        mock_test_suite,
        drift_client,
        sample_tensor,
        sample_target,
    ):
        """Test drift check with default n_predictions value."""
        # Setup GCS mock
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.name = "predictions/prediction_1"
        mock_blob.updated = datetime.now()
        mock_bucket.list_blobs.return_value = [mock_blob]
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client

        # Setup torch.load mock
        mock_torch_load.return_value = {"image": sample_tensor, "prediction": 0}
        mock_stack.return_value = sample_tensor

        # Setup TestSuite mock
        mock_suite_instance = MagicMock()
        mock_suite_instance.as_dict.return_value = {"summary": {"all_passed": True}}
        mock_test_suite.return_value = mock_suite_instance

        with patch("torch.load") as torch_load_patch:
            torch_load_patch.side_effect = [
                sample_tensor,  # train_images
                sample_target,  # train_targets
                {"image": sample_tensor, "prediction": 0},  # prediction data
            ]

            # Call without n_predictions parameter
            response = drift_client.post("/drift_check", json={})

            assert response.status_code == 200
            assert response.json()["status"] == "completed"


class TestDriftCheckResponseModel:
    """Tests for the DriftCheckResponse Pydantic model."""

    def test_drift_check_response_creation(self):
        """Test DriftCheckResponse creation."""
        from fruit_and_vegetable_disease.drift_api import DriftCheckResponse

        response = DriftCheckResponse(
            status="completed",
            passed=True,
            message="Test message",
        )

        assert response.status == "completed"
        assert response.passed is True
        assert response.message == "Test message"

    def test_drift_check_response_validation(self):
        """Test DriftCheckResponse validation."""
        from fruit_and_vegetable_disease.drift_api import DriftCheckResponse

        response = DriftCheckResponse(status="error", passed=False, message="Error occurred")

        assert response.status == "error"
        assert response.passed is False
