from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from fruit_and_vegetable_disease.model import Model

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DRIFTED_DIR = PROJECT_ROOT / "data" / "drifted"


def _load_split(images_path: Path, targets_path: Path) -> TensorDataset:
    """Load a tensor dataset from saved image and target tensors."""
    images = torch.load(images_path)
    targets = torch.load(targets_path)
    return TensorDataset(images, targets)


def _make_dataloader(dataset: TensorDataset, batch_size: int) -> DataLoader:
    """Create a dataloader for inference."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _evaluate_split(model: Model, dataloader: DataLoader) -> Tuple[float, float]:
	"""Compute accuracy and macro F1 on a dataloader."""
	model.eval()
	all_preds = []
	all_targets = []
	with torch.no_grad():
		for images, targets in dataloader:
			# Expand single channel to 3 channels (RGB)
			if images.shape[1] == 1:
				images = images.repeat(1, 3, 1, 1)
			# Resize from 32x32 to 224x224
			images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
			images = images.to(DEVICE)
			targets = targets.to(DEVICE)
			logits = model(images)
			preds = logits.argmax(dim=1)
			all_preds.extend(preds.cpu().tolist())
			all_targets.extend(targets.cpu().tolist())
	accuracy = accuracy_score(all_targets, all_preds)
	f1 = f1_score(all_targets, all_preds, average="macro")
	return accuracy, f1


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate robustness on clean vs drifted data")
	parser.add_argument(
		"--model-path",
		default=str(PROJECT_ROOT / "models" / "model.pth"),
		help="Path to the trained model checkpoint",
	)
	parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
	args = parser.parse_args()

	model = Model(num_classes=2).to(DEVICE)
	state_dict = torch.load(args.model_path, map_location=DEVICE)
	model.load_state_dict(state_dict)

	clean_ds = _load_split(PROCESSED_DIR / "test_images.pt", PROCESSED_DIR / "test_target.pt")
	drifted_ds = _load_split(DRIFTED_DIR / "drifted_test_images.pt", DRIFTED_DIR / "drifted_test_target.pt")

	clean_loader = _make_dataloader(clean_ds, args.batch_size)
	drifted_loader = _make_dataloader(drifted_ds, args.batch_size)

	print("Evaluating on clean test set...")
	clean_acc, clean_f1 = _evaluate_split(model, clean_loader)
	print(f"Clean accuracy: {clean_acc:.4f} | Clean macro F1: {clean_f1:.4f}")

	print("Evaluating on drifted test set...")
	drift_acc, drift_f1 = _evaluate_split(model, drifted_loader)
	print(f"Drifted accuracy: {drift_acc:.4f} | Drifted macro F1: {drift_f1:.4f}")

	delta_acc = clean_acc - drift_acc
	delta_f1 = clean_f1 - drift_f1
	print(f"Delta accuracy (clean - drifted): {delta_acc:.4f}")
	print(f"Delta macro F1 (clean - drifted): {delta_f1:.4f}")


if __name__ == "__main__":
    main()
