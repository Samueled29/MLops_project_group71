from fruit_and_vegetable_disease.data import create_datasets
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def test_create_datasets():
    train_set, test_set = create_datasets(PROCESSED_DATA_DIR)
    assert len(train_set) > 0
    assert len(test_set) > 0
