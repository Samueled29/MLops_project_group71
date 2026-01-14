import pytest
import torch
from fruit_and_vegetable_disease.model import Model


@pytest.mark.parametrize("batch_size,num_classes",[(1, 2),(2, 5),])
def test_model_output_shape(batch_size, num_classes):
    """Test if model returns logits with correct output shape."""
    model = Model(num_classes=num_classes, pretrained=False) # untrained model for testing to avoid downloading weights
    x = torch.rand(batch_size, 3, 224, 224)
    y = model(x)

    assert y.shape == (batch_size, num_classes), "Output tensor should have shape (batch_size, num_classes)"


@pytest.mark.parametrize("batch_size", [1, 3])
def test_model_output_is_tensor(batch_size):
    """Test if model output is a torch Tensor."""
    model = Model(num_classes=2, pretrained=False)
    x = torch.rand(batch_size, 3, 224, 224)
    y = model(x)

    assert isinstance(y, torch.Tensor), "Model output should be a torch Tensor"


@pytest.mark.parametrize("channels", [1, 3])
def test_model_accepts_correct_channel_count(channels):
    """Test if input tensor has expected number of channels."""
    model = Model(num_classes=2, pretrained=False)
    x = torch.rand(1, channels, 224, 224)

    if channels == 3:
        y = model(x)
        assert y.shape == (1, 2), "Model should run with 3-channel input"
    else:
        with pytest.raises(Exception):
            _ = model(x)
