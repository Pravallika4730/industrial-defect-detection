from pathlib import Path

import cv2
import torch
from torchvision import transforms


# ==========================================
# 1. PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "mvtec_anomaly_detection"
    / "bottle"
    / "train"
    / "good"
    / "000.png"
)


# ==========================================
# 2. LOAD IMAGE
# ==========================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")


print("=" * 60)
print("IMAGE PREPROCESSING TEST")
print("=" * 60)

print(f"\nImage path:")
print(IMAGE_PATH)

print(f"\nOriginal image shape: {image.shape}")


# ==========================================
# 3. CONVERT BGR → RGB
# ==========================================

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ==========================================
# 4. DEFINE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ==========================================
# 5. APPLY TRANSFORMATION
# ==========================================

tensor = transform(image)


# ==========================================
# 6. PRINT RESULT
# ==========================================

print(f"\nProcessed tensor shape: {tensor.shape}")

print(f"Tensor data type: {tensor.dtype}")

print(
    f"Pixel value range: "
    f"{tensor.min().item():.4f} - {tensor.max().item():.4f}"
)


# ==========================================
# 7. ADD BATCH DIMENSION
# ==========================================

batch = tensor.unsqueeze(0)

print(f"\nBatch shape: {batch.shape}")

print("\nPreprocessing completed successfully! ✅")