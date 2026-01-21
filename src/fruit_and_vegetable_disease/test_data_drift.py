from pathlib import Path
import torch
import pandas as pd

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import (
	DataDriftPreset,
	DataQualityPreset,
	TargetDriftPreset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DRIFTED_DIR = PROJECT_ROOT / "data" / "drifted"
REPORTS_DIR = PROJECT_ROOT / "reports" / "evidently"


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

	print("Generating detailed drift report...")
	report = Report(
		metrics=[
			DataDriftPreset(),
			DataQualityPreset(),
			TargetDriftPreset(),
		]
	)
	report.run(reference_data=reference_data, current_data=current_data)

	REPORTS_DIR.mkdir(parents=True, exist_ok=True)
	out_path = REPORTS_DIR / "data_drift_report.html"
	report.save_html(str(out_path))
	print(f"✓ Saved data drift report to: {out_path}")


if __name__ == "__main__":
	main()