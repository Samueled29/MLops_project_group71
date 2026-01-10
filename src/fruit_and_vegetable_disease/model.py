import torch
from torch import nn
from transformers import ViTForImageClassification, ViTImageProcessor


class Model(nn.Module):
    """Vision Transformer model for fruit and vegetable disease classification."""

    def __init__(self, num_classes: int = 1000, pretrained: bool = True):
        """Initialize the ViT model.

        Args:
            num_classes: Number of output classes for classification.
            pretrained: Whether to load pretrained weights.
        """
        super().__init__()
        model_name = "google/vit-base-patch16-224"

        if pretrained:
            self.model = ViTForImageClassification.from_pretrained(
                model_name, num_labels=num_classes, ignore_mismatched_sizes=True
            )
        else:
            from transformers import ViTConfig

            config = ViTConfig.from_pretrained(model_name, num_labels=num_classes)
            self.model = ViTForImageClassification(config)

        self.processor = ViTImageProcessor.from_pretrained(model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Model logits of shape (batch_size, num_classes).
        """
        outputs = self.model(pixel_values=x)
        return outputs.logits


if __name__ == "__main__":
    model = Model(num_classes=10)
    x = torch.rand(1, 3, 224, 224)
    print(f"Output shape of model: {model(x).shape}")
