import torch
import os
import shutil
from pathlib import Path
from transformers import ViTImageProcessorFast
from PIL import Image
from sklearn.model_selection import train_test_split

# DATA_URL= "https://huggingface.co/datasets/zolen/fruit_and_vegetable_disease_kaggle_mirror/resolve/main/apple_data.tar.gz"

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

image_processor = ViTImageProcessorFast.from_pretrained("google/vit-base-patch16-224")


def create_data_dir_structure() -> None:
    """Create data directory structure."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def download_and_extract_data(target_dir: str, remove_archive: bool = True) -> None:
    """Download tar.gz files from GCP using DVC and extract them."""
    print("Downloading raw data using DVC...")
    # Try uv first, go back to regular dvc if uv is not available
    download_cmd = "uv run dvc pull" if shutil.which("uv") else "dvc pull"
    download_exit_code = os.system(download_cmd)
    if download_exit_code != 0:
        print("Error: raw data download failed.")
        return

    # Extract both tar.gz files
    tar_files = ["Apple__Healthy.tar.gz", "Apple__Rotten.tar.gz"]

    for tar_file in tar_files:
        archive_path = os.path.join(target_dir, tar_file)
        if not os.path.exists(archive_path):
            print(f"Warning: {tar_file} not found in {target_dir}")
            continue

        extract_cmd = f"tar -xzf {archive_path} -C {target_dir} > /dev/null 2>&1"
        extract_exit_code = os.system(extract_cmd)
        if extract_exit_code == 0:
            print(f"Raw data correctly extracted from {tar_file} in {target_dir}")
            if remove_archive:
                os.remove(archive_path)
        else:
            print(f"Error: extraction of {tar_file} failed.")


def load_images(raw_dir: str):
    """Load images and their labels from raw data directory."""
    images = []
    targets = []

    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing folder: {raw_dir}")

    valid_dirs = {p.name for p in raw_dir.iterdir() if p.is_dir()}  # take only directories

    for class_name, label in class_map.items():
        if class_name not in valid_dirs:
            continue

        class_dir = raw_dir / class_name

        for img_path in os.listdir(class_dir):
            if img_path.startswith("."):
                continue

            if os.path.splitext(img_path)[1].lower() not in {".png", ".jpg", ".jpeg"}:
                continue

            full_img_path = class_dir / img_path
            raw_img = Image.open(full_img_path).convert("RGB")
            inputs = image_processor(raw_img, return_tensors="pt")
            images.append(inputs["pixel_values"].squeeze(0))
            targets.append(label)

    images = torch.stack(images)  # (N, 1, H, W)
    targets = torch.tensor(targets)  # (N,)
    return images, targets


def split_data(images: torch.Tensor, targets: torch.Tensor) -> tuple:
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        images, targets, test_size=test_size, random_state=seed, stratify=targets
    )

    torch.save(X_train.squeeze(1), RAW_DATA_DIR / "train_images.pt")
    torch.save(y_train, RAW_DATA_DIR / "train_target.pt")
    torch.save(X_test.squeeze(1), RAW_DATA_DIR / "test_images.pt")
    torch.save(y_test, RAW_DATA_DIR / "test_target.pt")

    return X_train, X_test, y_train, y_test


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data and save it to processed directory."""
    train_images: torch.Tensor = torch.load(f"{raw_dir}/train_images.pt")
    train_target: torch.Tensor = torch.load(f"{raw_dir}/train_target.pt")

    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")

    train_target = train_target.long()
    test_target = test_target.long()

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
        download_and_extract_data(
            url="https://huggingface.co/datasets/zolen/fruit_and_vegetable_disease_kaggle_mirror/resolve/main/apple_data.tar.gz",
            target_dir=RAW_DATA_DIR,
        )
    images, targets = load_images(RAW_DATA_DIR)
    split_data(images, targets)
    preprocess_data(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    create_datasets(PROCESSED_DATA_DIR)
