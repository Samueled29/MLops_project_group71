import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf

from fruit_and_vegetable_disease.data import PROCESSED_DATA_DIR, create_datasets

# from transformers import ViTForImageClassification, ViTImageProcessor
from fruit_and_vegetable_disease.model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def resize_and_expand_channels(images: torch.Tensor) -> torch.Tensor:
    """Resize 32x32 grayscale to 224x224 RGB required by Vision Transformer."""

    resized = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
    rgb = resized.repeat(1, 3, 1, 1)
    return rgb


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Train a model on fruit and vegetable disease dataset."""
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    train_set, _ = create_datasets(str(PROCESSED_DATA_DIR))
    train_dataloader = torch.utils.data.DataLoader(train_set, cfg.experiments.batch_size, shuffle=True)

    model = Model(num_classes=2).to(DEVICE)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    statistics: Dict[str, List[float]] = {"train_loss": [], "train_accuracy": []}
    for epoch in range(cfg.experiments.epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(resize_and_expand_channels(img))
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")

    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")

    # minimal usage / sanity checks
    print(f"Dataset size: {len(train_set)}")
    print(f"Model: {model.__class__.__name__}")


if __name__ == "__main__":
    train()
