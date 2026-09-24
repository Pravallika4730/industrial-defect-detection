from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ResNet18FeatureExtractor


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "mvtec_anomaly_detection"
    / "bottle"
    / "train"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. IMAGE TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ==========================================
# 3. LOAD NORMAL TRAINING DATA
# ==========================================

train_dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 4. DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("NORMAL FEATURE MODEL")
print("=" * 60)

print(f"\nDevice: {device}")
print(f"Training images: {len(train_dataset)}")


# ==========================================
# 5. LOAD PRETRAINED RESNET18
# ==========================================

model = ResNet18FeatureExtractor()

model = model.to(device)

model.eval()


# ==========================================
# 6. EXTRACT FEATURES
# ==========================================

all_features = []

print("\nExtracting features...")

with torch.no_grad():

    for images, _ in train_loader:

        images = images.to(device)

        features = model(images)

        all_features.append(features.cpu())


# ==========================================
# 7. COMBINE FEATURES
# ==========================================

all_features = torch.cat(all_features, dim=0)

print(f"\nFeature matrix shape:")
print(all_features.shape)


# ==========================================
# 8. CALCULATE NORMAL FEATURE CENTER
# ==========================================

normal_center = all_features.mean(dim=0)

print(f"\nNormal center shape:")
print(normal_center.shape)


# ==========================================
# 9. CALCULATE TRAINING DISTANCES
# ==========================================

distances = torch.norm(
    all_features - normal_center,
    dim=1
)

print("\nTraining feature distances:")

print(f"Minimum: {distances.min().item():.4f}")
print(f"Maximum: {distances.max().item():.4f}")
print(f"Mean:    {distances.mean().item():.4f}")


# ==========================================
# 10. SAVE NORMAL CENTER
# ==========================================

center_path = MODEL_DIR / "normal_center.pt"

torch.save(normal_center, center_path)

print("\nNormal feature center saved to:")

print(center_path)

print("\nNormal feature model created successfully! ✅")