import pytest
import torch
from fruit_and_vegetable_disease.train import resize_and_expand_channels


@pytest.mark.parametrize(
    "batch_size,height,width",
    [
        (1, 32, 32),
        (2, 32, 32),
    ],
)
def test_resize_and_expand_channels_output_shape(batch_size, height, width):
    """Test that grayscale 32x32 images are resized to 224x224 RGB."""
    x = torch.rand(batch_size, 1, height, width)
    y = resize_and_expand_channels(x)

    assert y.shape == (batch_size, 3, 224, 224), "Output should be batch_size x 3 x 224 x 224"


@pytest.mark.parametrize("batch_size", [1, 2])
def test_resize_and_expand_channels_dtype(batch_size):
    """Test that output tensor dtype is same as input."""
    x = torch.rand(batch_size, 1, 32, 32)
    y = resize_and_expand_channels(x)

    assert y.dtype == x.dtype, "Output tensor should have same dtype as input"


def test_accuracy_calculation():
    """Test that accuracy calculation produces a float between 0 and 1."""
    y_pred = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    target = torch.tensor([1, 0])

    accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()

    assert 0.0 <= accuracy <= 1.0, "Accuracy should be a float between 0 and 1"
