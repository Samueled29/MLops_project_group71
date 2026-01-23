import matplotlib.pyplot as plt
import torch
import hydra
import wandb
from pathlib import Path
from typing import Dict, List
from omegaconf import DictConfig, OmegaConf

from torch.profiler import (
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
    profile,
)

from fruit_and_vegetable_disease.model import Model
from fruit_and_vegetable_disease.data import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DATA_URL,
    download_and_extract_data,
    load_images,
    split_data,
    preprocess_data,
    create_datasets,
    is_processed_data_valid,
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for img, target in dataloader:
            img, target = img.to(device), target.to(device)
            logits = model(img)
            loss = loss_fn(logits, target)

            total_loss += loss.item() * target.size(0)
            correct += (logits.argmax(dim=1) == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / total if total else 0.0
    acc = correct / total if total else 0.0
    return avg_loss, acc


def ensure_data_ready() -> None:

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    required = ["train_images.pt", "train_target.pt", "test_images.pt", "test_target.pt"]
    processed_ok = all((PROCESSED_DATA_DIR / f).exists() for f in required)

    if processed_ok:
        return

    
    if not any(RAW_DATA_DIR.iterdir()):
        download_and_extract_data(url=DATA_URL, target_dir=str(RAW_DATA_DIR))

    
    images, targets = load_images(str(RAW_DATA_DIR))
    split_data(images, targets)
    preprocess_data(str(RAW_DATA_DIR), str(PROCESSED_DATA_DIR))


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=cfg.wandb.run_name,
        tags=getattr(cfg.wandb, "tags", None),
        notes=getattr(cfg.wandb, "notes", None),
        mode=getattr(cfg.wandb, "mode", "online"),
        reinit=True,
    )


    ensure_data_ready()
    train_set, test_set = create_datasets(str(PROCESSED_DATA_DIR))

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.experiments.batch_size,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.experiments.batch_size,
        shuffle=False,
    )

    print("DATA SETUP COMPLETE")

    
    
    model = hydra.utils.instantiate(cfg.model, num_classes=cfg.dataset.num_classes).to(DEVICE)
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
        with_stack=False,
    )

    statistics: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    prof.start()


    for epoch in range(cfg.experiments.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_acc_sum = 0.0
        num_batches = 0

        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(img)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()

            batch_loss = float(loss.item())
            batch_acc = float((logits.argmax(dim=1) == target).float().mean().item())

            statistics["train_loss"].append(batch_loss)
            statistics["train_accuracy"].append(batch_acc)

            epoch_loss_sum += batch_loss
            epoch_acc_sum += batch_acc
            num_batches += 1

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {batch_loss:.4f}, acc: {batch_acc:.4f}")
                wandb.log(
                    {
                        "train/batch_loss": batch_loss,
                        "train/batch_accuracy": batch_acc,
                        "epoch": epoch,
                        "batch": i,
                    }
                )

            prof.step()


        test_loss, test_acc = evaluate(model, test_dataloader, loss_fn, DEVICE)
        statistics["test_loss"].append(test_loss)
        statistics["test_accuracy"].append(test_acc)

        avg_epoch_loss = epoch_loss_sum / num_batches if num_batches else 0.0
        avg_epoch_acc = epoch_acc_sum / num_batches if num_batches else 0.0

        print(f"Epoch {epoch} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

        wandb.log(
            {
                "train/epoch_loss": avg_epoch_loss,
                "train/epoch_accuracy": avg_epoch_acc,
                "test/loss": test_loss,
                "test/accuracy": test_acc,
                "epoch": epoch,
            }
        )

    prof.stop()
    print("Training complete")


    try:
        print("\n=== Profiling Summary ===")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        if torch.cuda.is_available():
            print("\n=== Memory Usage ===")
            print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    except Exception:
        pass


    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), "models/model.pth")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"], label="train")
    axs[0].plot(
        [None] * (len(statistics["train_loss"]) - len(statistics["test_loss"])) + statistics["test_loss"],
        label="test",
    )
    axs[0].set_title("Loss")
    axs[0].legend()

    axs[1].plot(statistics["train_accuracy"], label="train")
    axs[1].plot(
        [None] * (len(statistics["train_accuracy"]) - len(statistics["test_accuracy"])) + statistics["test_accuracy"],
        label="test",
    )
    axs[1].set_title("Accuracy")
    axs[1].legend()

    fig.savefig("reports/figures/training_statistics.png")
    plt.close(fig)

    wandb.log({"training_statistics": wandb.Image("reports/figures/training_statistics.png")})
    wandb.finish()

    print(f"Train size: {len(train_set)} | Test size: {len(test_set)}")
    print(f"Model: {model.__class__.__name__}")


if __name__ == "__main__":
    train()

