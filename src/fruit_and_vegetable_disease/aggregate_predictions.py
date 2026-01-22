from pathlib import Path
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TENSOR_DIR = PROJECT_ROOT / "logs" / "predictions" / "tensors"
OUTPUT_DIR = PROJECT_ROOT / "data" / "production"


def aggregate_tensors() -> None:
    """Load individual tensor files and batch them into production_images.pt."""
    if not TENSOR_DIR.exists():
        print(f"✗ Tensor directory not found: {TENSOR_DIR}")
        return

    tensor_files = sorted(TENSOR_DIR.glob("*.pt"))
    if not tensor_files:
        print(f"✗ No tensor files found in: {TENSOR_DIR}")
        return

    print(f"Found {len(tensor_files)} tensor files. Aggregating...")

    # Load all tensors (each is shape [1, 3, 224, 224])
    all_tensors = []
    for tensor_file in tensor_files:
        tensor = torch.load(tensor_file, map_location="cpu")
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        all_tensors.append(tensor)

    if not all_tensors:
        print("✗ No tensors to aggregate")
        return

    # Stack into batch [N, 3, 224, 224]
    production_images = torch.stack(all_tensors, dim=0)
    # Create dummy targets (0 = healthy) for drift detection
    production_targets = torch.zeros(len(all_tensors), dtype=torch.long)

    # Save aggregated tensors
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images_path = OUTPUT_DIR / "production_images.pt"
    targets_path = OUTPUT_DIR / "production_target.pt"

    torch.save(production_images, images_path)
    torch.save(production_targets, targets_path)

    print(f"✓ Saved {len(all_tensors)} aggregated tensors:")
    print(f"  Images: {images_path} (shape: {production_images.shape})")
    print(f"  Targets: {targets_path} (shape: {production_targets.shape})")


if __name__ == "__main__":
    aggregate_tensors()
