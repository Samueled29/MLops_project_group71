import pytest
import os
import torch
import math
from fruit_and_vegetable_disease.data import test_size, class_map
from fruit_and_vegetable_disease.data import load_images, normalize, split_data, create_datasets
from tests import _PATH_RAW_DATA, _PATH_PROCESSED_DATA


def test_data_directories_exist():
    """Test if raw and processed data directories exist."""
    assert os.path.isdir(_PATH_RAW_DATA), f"Raw data directory {_PATH_RAW_DATA} does not exist."
    assert os.path.isdir(_PATH_PROCESSED_DATA), f"Processed data directory {_PATH_PROCESSED_DATA} does not exist."


@pytest.mark.skipif(
    not os.path.exists(_PATH_RAW_DATA) or len(os.listdir(_PATH_RAW_DATA)) == 0,
    reason="Data files not found"
)
def test_download_data():
    """Test if data download function works correctly."""
    # This test assumes that the data download function is called elsewhere
    # and that the raw data directory is populated.
    assert len(os.listdir(_PATH_RAW_DATA)) > 0, f"Raw data directory {_PATH_RAW_DATA} is empty."


@pytest.mark.skipif(
    not os.path.exists(_PATH_RAW_DATA) or len(os.listdir(_PATH_RAW_DATA)) == 0,
    reason="Data files not found"
)
def test_img_grayscale_tensor():
    """Test if loaded images are grayscale tensors."""
    images, _ = load_images(raw_dir=_PATH_RAW_DATA)
    assert images.shape[1] == 1, "The images should be grayscale (1 channel)"


def test_load_images_folder():
    """Test if loading from a non-existent folder raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match=r"^Missing folder: .+$"):
        load_images(raw_dir="non_existent_folder")


@pytest.mark.skipif(
    not os.path.exists(_PATH_RAW_DATA) or len(os.listdir(_PATH_RAW_DATA)) == 0,
    reason="Data files not found"
)
def test_load_images_output_shape():
    """Test if loaded images and targets have correct dimensions."""
    images, targets = load_images(raw_dir=_PATH_RAW_DATA)
    assert images.ndim == 4, "The images should have 4 dimensions (N, 1, H, W)"
    assert targets.ndim == 1, "The targets should have 1 dimension (N,)"
    assert images.size(0) == targets.size(0), "The number of images should match the number of targets"


def test_size_testset():
    """Test if test_size is a float between 0 and 1."""
    assert isinstance(test_size, float), "test_size should be a float"
    assert test_size > 0 and test_size < 1, "test_size should be between 0 and 1"


@pytest.mark.skipif(
    not os.path.exists(_PATH_RAW_DATA) or len(os.listdir(_PATH_RAW_DATA)) == 0,
    reason="Data files not found"
)
def test_classmap_labels():
    """Test if that the unique values in the targets match the values from class_map."""
    _, targets = load_images(raw_dir=_PATH_RAW_DATA)
    unique_labels = set(int(x) for x in targets.tolist())

    # Convert class_map values to integers when possible to compare with unique_labels
    try:
        map_vals = set(int(v) for v in class_map.values())
    except Exception:
        map_vals = set(class_map.values())

    assert unique_labels == map_vals, "Unique labels found in targets must exactly match the set of values in class_map"


@pytest.mark.skipif(
    not os.path.exists(_PATH_RAW_DATA) or len(os.listdir(_PATH_RAW_DATA)) == 0,
    reason="Data files not found"
)
def test_normalize_img():
    """Test if the normalize function standardizes the images correctly."""
    images, _ = load_images(raw_dir=_PATH_RAW_DATA)

    normalized_images = normalize(images)
    mean = normalized_images.mean().item()
    std = normalized_images.std().item()

    assert abs(mean) < 1e-5, "Mean of normalized images should be approximately 0"
    assert abs(std - 1) < 1e-5, "Standard deviation of normalized images should be approximately 1"


def test_normalize_empty_tensor():
    """Test if the normalize function handles an empty tensor without errors."""
    empty_images = torch.empty((0, 1, 32, 32))
    normalized_images = normalize(empty_images)
    assert normalized_images.shape == empty_images.shape, "Normalized empty tensor should have the same shape as input"


@pytest.mark.skipif(
    not os.path.exists(_PATH_RAW_DATA) or len(os.listdir(_PATH_RAW_DATA)) == 0,
    reason="Data files not found"
)
def test_split_data_sizes():
    """Test split_data function splits datasets into correct sizes and preserves all samples."""

    images, targets = load_images(raw_dir=_PATH_RAW_DATA)
    n_samples = images.size(0)

    train_images, test_images, train_targets, test_targets = split_data(images, targets)

    n_test_expected = math.ceil(n_samples * test_size)  # sklearn's train_test_split uses ceil for test size
    n_train_expected = n_samples - n_test_expected

    assert train_images.size(0) == n_train_expected, "Number of training samples is incorrect"
    assert test_images.size(0) == n_test_expected, "Number of test samples is incorrect"
    assert train_targets.size(0) == n_train_expected, "Number of training targets is incorrect"
    assert test_targets.size(0) == n_test_expected, "Number of test targets is incorrect"

    assert train_images.ndim == 4 and test_images.ndim == 4, "Images should have 4 dimensions (N, C, H, W)"
    assert train_targets.ndim == 1 and test_targets.ndim == 1, "Targets should have 1 dimension (N,)"

    # Ensure that concatenating train and test targets reproduces the original set of targets
    combined = torch.cat([train_targets, test_targets])
    orig_sorted, _ = torch.sort(targets)
    combined_sorted, _ = torch.sort(combined)
    assert torch.equal(orig_sorted, combined_sorted), "Split must preserve all original samples (targets mismatch)"


@pytest.mark.skipif(
    not os.path.exists(_PATH_PROCESSED_DATA) or len(os.listdir(_PATH_PROCESSED_DATA)) == 0,
    reason="Data files not found"
)
def test_processed_data_files_exist():
    """Test if the processed data files exist after preprocessing."""
    processed_files = ["train_images.pt", "train_target.pt", "test_images.pt", "test_target.pt"]
    for file_name in processed_files:
        file_path = os.path.join(_PATH_PROCESSED_DATA, file_name)
        assert os.path.isfile(file_path), f"Processed data file {file_name} should exist in {_PATH_PROCESSED_DATA}"


@pytest.mark.skipif(
    not os.path.exists(_PATH_PROCESSED_DATA) or len(os.listdir(_PATH_PROCESSED_DATA)) == 0,
    reason="Data files not found"
)
def test_create_tensor_datasets():
    """Test if create_datasets function returns valid datasets."""

    train_set, test_set = create_datasets(processed_dir=_PATH_PROCESSED_DATA)

    assert isinstance(
        train_set, torch.utils.data.Dataset
    ), "train_set should be an instance of torch.utils.data.Dataset"
    assert isinstance(test_set, torch.utils.data.Dataset), "test_set should be an instance of torch.utils.data.Dataset"

    assert len(train_set) > 0, "train_set should not be empty"
    assert len(test_set) > 0, "test_set should not be empty"
