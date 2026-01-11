import torch
import typer

from fruit_and_vegetable_disease.data import PROCESSED_DATA_DIR, create_datasets
from fruit_and_vegetable_disease.model import Model
from fruit_and_vegetable_disease.train import resize_and_expand_channels

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str, batch_size: int = 32) -> None:
    """Evaluate the model."""
    print(f"Evaluating model: {model_checkpoint}")
    print(f"Batch size: {batch_size}")

    model = Model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = create_datasets(str(PROCESSED_DATA_DIR))
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for img, target in test_dataloader:
            img = resize_and_expand_channels(img)
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    typer.run(evaluate)
