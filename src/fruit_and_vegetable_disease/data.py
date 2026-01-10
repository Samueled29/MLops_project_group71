import torch
import os
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import requests
import tarfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

test_size = 0.2
seed = 42

class_map = {
    "Apple__Healthy": 0,
    "Apple__Rotten": 1,
}

def download_and_extract_data(url: str, target_dir: str, archive_name: str = "apple_data.tar.gz",remove_archive: bool = True):
    """Download a tar.gz archive and extract it to target_dir."""

    os.makedirs(target_dir, exist_ok=True)
    archive_path = os.path.join(target_dir, archive_name)

    print("Downloading dataset...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print("Extracting dataset...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir)

    if remove_archive:
        os.remove(archive_path)

    print("Dataset ready at:", target_dir)

def pil_to_tensor_grayscale(img: Image.Image, size: tuple[int, int] = (32, 32)) -> torch.Tensor:
    """Convert PIL image to grayscale tensor (1, H, W) in [0, 1], resized."""
    img = img.convert("L")
    img = img.resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def load_images():
    """Load images and their labels from raw data directory."""
    images = []
    targets = []

    for class_name, label in class_map.items():
        class_dir = RAW_DATA_DIR / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing folder: {class_dir}")

        for img_path in class_dir.iterdir():
            if img_path.name.startswith("."):
                continue

            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue

            img = Image.open(img_path).convert("RGB")
            images.append(pil_to_tensor_grayscale(img))
            targets.append(label)

    images = torch.stack(images)        # (N, 1, H, W)
    targets = torch.tensor(targets)     # (N,)
    return images, targets

def split_data(images: torch.Tensor, targets: torch.Tensor) -> tuple:
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=test_size, random_state=seed, stratify=targets)

    torch.save(X_train.squeeze(1), RAW_DATA_DIR / "train_images.pt")
    torch.save(y_train, RAW_DATA_DIR / "train_target.pt")
    torch.save(X_test.squeeze(1), RAW_DATA_DIR / "test_images.pt")
    torch.save(y_test, RAW_DATA_DIR / "test_target.pt")

def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()

def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data and save it to processed directory."""
    train_images: torch.Tensor = torch.load(f"{raw_dir}/train_images.pt")
    train_target: torch.Tensor = torch.load(f"{raw_dir}/train_target.pt")

    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")

def create_datasets(processed_dir: str) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for the main dataset."""
    train_images = torch.load(f"{processed_dir}/train_images.pt")
    train_target = torch.load(f"{processed_dir}/train_target.pt")
    test_images = torch.load(f"{processed_dir}/test_images.pt")
    test_target = torch.load(f"{processed_dir}/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    # execute the data download only if raw_data_dir is empty
    if not RAW_DATA_DIR.exists() or not any(RAW_DATA_DIR.iterdir()):
        download_and_extract_data(url="https://huggingface.co/datasets/zolen/fruit_and_vegetable_disease_kaggle_mirror/resolve/main/apple_data.tar.gz", target_dir=RAW_DATA_DIR)
    images, targets = load_images()
    split_data(images, targets)
    preprocess_data(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    create_datasets(PROCESSED_DATA_DIR)
    

