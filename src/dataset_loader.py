from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
)


# ==========================================
# 2. IMAGE TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ==========================================
# 3. CREATE DATASET
# ==========================================

train_dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=transform
)


# ==========================================
# 4. CREATE DATALOADER
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 5. TEST DATALOADER
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("MVTec Bottle DataLoader Test")
    print("=" * 60)

    print(f"\nDataset size: {len(train_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    images, labels = next(iter(train_loader))

    print(f"\nBatch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")

    print(f"\nLabels in first batch: {labels.tolist()}")

    print("\nDataLoader test completed successfully! ✅")