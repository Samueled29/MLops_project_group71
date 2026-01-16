import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
<<<<<<< Updated upstream
from typing import Dict, List

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

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

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=cfg.wandb.run_name,
        reinit=True,
    )

    train_set, _ = create_datasets(str(PROCESSED_DATA_DIR))
    train_dataloader = torch.utils.data.DataLoader(train_set, cfg.experiments.batch_size, shuffle=True)

    model = Model(num_classes=2).to(DEVICE)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    profiler_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    prof = profile(
        activities=activities,
        schedule=profiler_schedule,
        on_trace_ready=tensorboard_trace_handler("./logs/profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # Disable stack traces to save memory
    )

    statistics: Dict[str, List[float]] = {"train_loss": [], "train_accuracy": []}
    prof.start()

    for epoch in range(cfg.experiments.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        for i, (img, target) in enumerate(train_dataloader):
=======
import typer

#from transformers import ViTForImageClassification, ViTImageProcessor

from fruit_and_vegetable_disease.model import Model
from fruit_and_vegetable_disease.data import create_datasets, PROCESSED_DATA_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def resize_and_expand_channels(images: torch.Tensor) -> torch.Tensor:
    """Resize 32x32 grayscale to 224x224 RGB required by Vision Transformer."""

    resized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
    rgb = resized.repeat(1, 3, 1, 1)
    return rgb


def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on fruit and vegetable disease dataset."""

    print(f"{lr=}, {batch_size=}, {epochs=}")

    train_set, _ = create_datasets(str(PROCESSED_DATA_DIR))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model = Model(num_classes=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img,target) in enumerate(train_dataloader):
>>>>>>> Stashed changes
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(resize_and_expand_channels(img))
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
<<<<<<< Updated upstream

            batch_loss = loss.item()
            batch_accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()

            statistics["train_loss"].append(batch_loss)
            statistics["train_accuracy"].append(batch_accuracy)
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
            num_batches += 1

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {batch_loss:.4f}, accuracy: {batch_accuracy:.4f}")
                wandb.log(
                    {
                        "train/batch_loss": batch_loss,
                        "train/batch_accuracy": batch_accuracy,
                        "train/epoch": epoch,
                        "train/batch": i,
                    }
                )

            prof.step()

        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_accuracy = epoch_accuracy / num_batches
        print(f"Epoch {epoch} complete - Avg Loss: {avg_epoch_loss:.4f}, " f"Avg Accuracy: {avg_epoch_accuracy:.4f}")

        wandb.log(
            {
                "train/epoch_loss": avg_epoch_loss,
                "train/epoch_accuracy": avg_epoch_accuracy,
                "epoch": epoch,
            }
        )

    prof.stop()
    print("Training complete")

    # Print profiling summary
    print("\n=== Profiling Summary ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    if torch.cuda.is_available():
        print("\n=== Memory Usage ===")
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

=======
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
   
    print("Training complete")

>>>>>>> Stashed changes
    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")
<<<<<<< Updated upstream

    wandb.log({"training_statistics": wandb.Image("reports/figures/training_statistics.png")})
=======
>>>>>>> Stashed changes

    # minimal usage / sanity checks
    print(f"Dataset size: {len(train_set)}")
    print(f"Model: {model.__class__.__name__}")

    wandb.finish()


if __name__ == "__main__":
    typer.run(train)

