from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import ResNet18FeatureExtractor


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "mvtec_anomaly_detection"
    / "bottle"
)

MODEL_DIR = PROJECT_ROOT / "models"

LOCAL_STATS_PATH = MODEL_DIR / "local_feature_stats.pt"


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Improved Local Feature Anomaly Scoring")
print("=" * 60)

print(f"\nDevice: {device}")


# ============================================================
# MODEL
# ============================================================

model = ResNet18FeatureExtractor()
model = model.to(device)
model.eval()


# ============================================================
# LOAD LOCAL FEATURE STATISTICS
# ============================================================

stats = torch.load(
    LOCAL_STATS_PATH,
    map_location=device,
    weights_only=True
)

normal_mean = stats["mean"].to(device)
normal_std = stats["std"].to(device)

print(f"\nNormal mean shape: {normal_mean.shape}")
print(f"Normal std shape : {normal_std.shape}")


# ============================================================
# ANOMALY SCORE FUNCTION
# ============================================================

def calculate_anomaly_score(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        feature_map = model.features(image_tensor)

        # Convert feature map to local standardized anomaly values
        standardized_difference = (
            feature_map - normal_mean.unsqueeze(0)
        ) / (normal_std.unsqueeze(0) + 1e-6)

        # Absolute deviation
        anomaly_map = torch.abs(
            standardized_difference
        )

        # Average across feature channels
        spatial_map = anomaly_map.mean(dim=1)

        # Overall anomaly score
        score = spatial_map.mean()

    return score.item(), spatial_map.squeeze(0)


# ============================================================
# TEST IMAGES
# ============================================================

test_images = [
    DATASET_DIR / "test" / "good" / "000.png",
    DATASET_DIR / "test" / "broken_large" / "000.png",
    DATASET_DIR / "test" / "broken_small" / "000.png",
    DATASET_DIR / "test" / "contamination" / "000.png",
]


# ============================================================
# RUN TEST
# ============================================================

print("\n" + "=" * 60)
print("ANOMALY SCORE COMPARISON")
print("=" * 60)

for image_path in test_images:

    score, anomaly_map = calculate_anomaly_score(
        image_path
    )

    print(
        f"\n{image_path.parent.name:20s}"
        f" Score: {score:.4f}"
    )

    print(
        f"Feature anomaly map: "
        f"{tuple(anomaly_map.shape)}"
    )


print("\n" + "=" * 60)
print("IMPROVED ANOMALY SCORING COMPLETED! ✅")
print("=" * 60)