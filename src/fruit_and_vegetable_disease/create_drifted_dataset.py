import torch
import os
from torchvision import datasets, transforms

INPUT_IMAGES = "data/processed/test_images.pt"
INPUT_TARGETS = "data/processed/test_target.pt"

OUTPUT_DIR = "data/drifted"
OUTPUT_IMAGES = os.path.join(OUTPUT_DIR, "drifted_test_images.pt")
OUTPUT_TARGETS = os.path.join(OUTPUT_DIR, "drifted_test_target.pt")

# Create output folder if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the original test dataset
images = torch.load(INPUT_IMAGES)
targets = torch.load(INPUT_TARGETS)

print("Shape images: ", images.shape)
print("Shape targets: ", targets.shape)

# Introduce drift

drift_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.GaussianBlur(5),
    transforms.RandomRotation(15),
])

# Apply drift transformation to each image
drifted_images = []

for img in images:
    drifted= drift_transform(img)
    drifted_images.append(drifted)

drifted_images = torch.stack(drifted_images)

torch.save(drifted_images, OUTPUT_IMAGES)
torch.save(targets, OUTPUT_TARGETS)

print(f"Drifted dataset saved to {OUTPUT_DIR}")

