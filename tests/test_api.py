import io
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.radio_check_app import app, load_all_models_and_assets


@pytest.fixture(autouse=True)
def env_setup():
    """Ensure environment variables are set before any app code runs."""
    os.environ["MLFLOW_URI"] = "http://localhost:5000"
    os.environ["S3_MONITORING_BUCKET"] = "test-bucket"


@pytest.fixture
def mock_external_services():
    """Mock external model loads, boto3, MLflow, and model inference tensors."""
    with (
        patch("src.radio_check_app.AutoTokenizer.from_pretrained") as mock_tokenizer,
        patch("src.radio_check_app.AutoModel.from_pretrained") as mock_text_model,
        patch("src.radio_check_app.get_biovil_t_image_encoder") as mock_img_encoder,
        patch("src.radio_check_app.mlflow.pytorch.load_model") as mock_mlflow_load,
        patch("src.radio_check_app.mlflow.set_tracking_uri"),
        patch("src.radio_check_app.boto3.client") as mock_boto,
    ):
        # 1. Mock S3 Client
        mock_s3_instance = MagicMock()
        mock_boto.return_value = mock_s3_instance

        # 2. Mock Text Model Output (returns 3D Tensor for [batch, seq_len, hidden_dim])
        mock_text_instance = MagicMock()
        mock_text_instance.return_value.last_hidden_state = torch.zeros((1, 512, 768))
        mock_text_model.return_value = mock_text_instance.to.return_value

        # 3. Mock Tokenizer Output
        mock_tok_instance = MagicMock()
        mock_tok_instance.return_value.input_ids = torch.zeros(
            (1, 512), dtype=torch.long
        )
        mock_tok_instance.return_value.attention_mask = torch.ones(
            (1, 512), dtype=torch.long
        )
        mock_tok_instance.to.return_value = mock_tok_instance.return_value
        mock_tokenizer.from_pretrained.return_value = mock_tok_instance

        # 4. Mock Image Model Output
        mock_img_instance = MagicMock()
        mock_img_instance.return_value.projected_patch_embeddings = torch.zeros(
            (1, 128, 512)
        )
        mock_img_encoder.return_value.to.return_value = mock_img_instance

        # 5. Mock Cross-Attention Classifier Output (logit layer output)
        mock_classifier = MagicMock()
        # Returns Tensor([2.0]) -> sigmoid(2.0) approx 0.8808
        mock_classifier.return_value = torch.tensor([[2.0]])
        mock_mlflow_load.return_value = mock_classifier

        # Manually load mocks into global variables
        load_all_models_and_assets()

        yield {
            "s3": mock_s3_instance,
            "classifier": mock_classifier,
            "mlflow_load": mock_mlflow_load,
        }


@pytest.fixture
def client(mock_external_services):
    """Provide TestClient instance without re-triggering real app startup events."""
    # Prevent FastAPI on_event("startup") from attempting live MLflow downloads
    with patch("src.radio_check_app.load_all_models_and_assets"):
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def dummy_image_bytes():
    """Create a minimal 10x10 dummy JPEG image in memory."""
    buf = io.BytesIO()
    image = Image.new("RGB", (10, 10), color="white")
    image.save(buf, format="JPEG")
    return buf.getvalue()


# -------------------------------------------------------------------
# Test Cases
# -------------------------------------------------------------------


def test_predict_endpoint_missing_input(client):
    """Test request failure when form payload is missing required image file."""
    response = client.post("/predict", data={"text_input": "Sample text"})
    assert response.status_code == 422


def test_reload_model_endpoint(client, mock_external_services):
    """Test successful model reloads via MLflow."""
    response = client.post("/reload-model")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "biovil_t_lead_demo@prod loaded successfully" in data["message"]


def test_reload_model_failure(client):
    """Test handling of MLflow failure during model reload."""
    with patch(
        "src.radio_check_app.mlflow.pytorch.load_model",
        side_effect=RuntimeError("MLflow connection error"),
    ):
        response = client.post("/reload-model")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "Fail to load the model" in data["message"]
