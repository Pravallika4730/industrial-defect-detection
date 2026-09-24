from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ResNet18FeatureExtractor


# ==========================================
# 1. PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "mvtec_anomaly_detection"
    / "bottle"
    / "train"
    / "good"
)


# ==========================================
# 2. IMAGE TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ==========================================
# 3. DATASET
# ==========================================

train_dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR.parent),
    transform=transform
)


# ==========================================
# 4. DATALOADER
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 5. DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("MVTec → ResNet18 Feature Extraction")
print("=" * 60)

print(f"\nDevice: {device}")

print(f"Dataset size: {len(train_dataset)}")


# ==========================================
# 6. CREATE MODEL
# ==========================================

model = ResNet18FeatureExtractor()

model = model.to(device)

model.eval()


# ==========================================
# 7. EXTRACT FEATURES FROM ONE BATCH
# ==========================================

images, labels = next(iter(train_loader))

images = images.to(device)


with torch.no_grad():

    features = model(images)


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print(f"\nInput batch shape:")
print(images.shape)

print(f"\nFeature batch shape:")
print(features.shape)

print("\nFeature extraction completed successfully! ✅")