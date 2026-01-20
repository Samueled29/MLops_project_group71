from pathlib import Path
import sys
import torch
import pandas as pd

from evidently.legacy.test_suite import TestSuite
from evidently.legacy.tests import (
	TestNumberOfMissingValues,
	TestShareOfDriftedColumns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DRIFTED_DIR = PROJECT_ROOT / "data" / "drifted"


def _ensure_nchw(x: torch.Tensor) -> torch.Tensor:
	if x.dim() == 3:
		return x.unsqueeze(1)
	return x


def _to_features_df(images: torch.Tensor, targets: torch.Tensor) -> pd.DataFrame:
	images = _ensure_nchw(images).float()
	flat = images.view(images.size(0), -1)
	df = pd.DataFrame(
		{
			"pixel_mean": flat.mean(dim=1).cpu().numpy(),
			"pixel_std": flat.std(dim=1, unbiased=False).cpu().numpy(),
			"label": targets.cpu().numpy(),
		}
	)
	return df


def main() -> None:
	ref_images = torch.load(PROCESSED_DIR / "train_images.pt")
	ref_targets = torch.load(PROCESSED_DIR / "train_target.pt")

	cur_images = torch.load(DRIFTED_DIR / "drifted_test_images.pt")
	cur_targets = torch.load(DRIFTED_DIR / "drifted_test_target.pt")

	reference_data = _to_features_df(ref_images, ref_targets)
	current_data = _to_features_df(cur_images, cur_targets)

	print("Running data quality tests...")
	test_suite = TestSuite(
		tests=[
			TestNumberOfMissingValues(),
			TestShareOfDriftedColumns(lt=0.5),  # Fail if >50% of features drifted
		]
	)
	test_suite.run(reference_data=reference_data, current_data=current_data)
	test_results = test_suite.as_dict()

	all_passed = test_results["summary"]["all_passed"]
	print(f"All tests passed: {all_passed}")

	if not all_passed:
		print("\n❌ Data quality tests failed!")
		sys.exit(1)
	else:
		print("\n✓ All data quality tests passed")


if __name__ == "__main__":
	main()
