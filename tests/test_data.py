from fruit_and_vegetable_disease.data import create_datasets


def test_create_datasets():
    train_set, test_set = create_datasets("data/processed")
    assert len(train_set) > 0
    assert len(test_set) > 0
