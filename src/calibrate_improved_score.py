from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

from model import ResNet18FeatureExtractor


# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "mvtec_anomaly_detection"
    / "bottle"
)

MODEL_DIR = PROJECT_ROOT / "models"

LOCAL_STATS_PATH = MODEL_DIR / "local_feature_stats.pt"
OUTPUT_PATH = MODEL_DIR / "improved_score_threshold.pt"


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# -----------------------------
# Device
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# -----------------------------
# Load ResNet18 feature extractor
# -----------------------------
model = ResNet18FeatureExtractor().to(device)
model.eval()


# -----------------------------
# Load local feature statistics
# -----------------------------
stats = torch.load(
    LOCAL_STATS_PATH,
    map_location=device,
    weights_only=True
)

normal_mean = stats["mean"].to(device)
normal_std = stats["std"].to(device)


# -----------------------------
# Calculate improved anomaly score
# -----------------------------
def calculate_anomaly_score(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        feature_map = model.features(image_tensor)

        standardized_difference = (
            feature_map
            - normal_mean.unsqueeze(0)
        ) / (
            normal_std.unsqueeze(0) + 1e-6
        )

        anomaly_map = torch.abs(
            standardized_difference
        )

        spatial_map = anomaly_map.mean(dim=1)

        score = spatial_map.mean()

    return score.item()


# -----------------------------
# Collect normal training images
# -----------------------------
train_good_dir = DATASET_DIR / "train" / "good"

image_paths = sorted(
    train_good_dir.glob("*.png")
)

print(f"Normal training images: {len(image_paths)}")

if len(image_paths) == 0:
    raise RuntimeError(
        "No training images found!"
    )


# -----------------------------
# Calculate scores
# -----------------------------
scores = []

print("\nCalculating anomaly scores...")

for index, image_path in enumerate(image_paths):

    score = calculate_anomaly_score(image_path)

    scores.append(score)

    if (index + 1) % 25 == 0:
        print(
            f"Processed {index + 1}/{len(image_paths)}"
        )


# -----------------------------
# Convert to tensor
# -----------------------------
scores_tensor = torch.tensor(scores)


# -----------------------------
# Statistics
# -----------------------------
minimum = scores_tensor.min().item()
maximum = scores_tensor.max().item()
mean = scores_tensor.mean().item()
std = scores_tensor.std().item()

percentile_95 = torch.quantile(
    scores_tensor, 0.95
).item()

percentile_99 = torch.quantile(
    scores_tensor, 0.99
).item()


# -----------------------------
# Use maximum normal score
# as conservative threshold
# -----------------------------
threshold = maximum


# -----------------------------
# Print results
# -----------------------------
print("\n" + "=" * 50)
print("IMPROVED ANOMALY SCORE CALIBRATION")
print("=" * 50)

print(f"Minimum score : {minimum:.4f}")
print(f"Maximum score : {maximum:.4f}")
print(f"Mean score    : {mean:.4f}")
print(f"Std deviation : {std:.4f}")
print(f"95th percentile: {percentile_95:.4f}")
print(f"99th percentile: {percentile_99:.4f}")

print("\nSelected threshold:")
print(f"{threshold:.4f}")


# -----------------------------
# Save threshold
# -----------------------------
torch.save(
    {
        "threshold": threshold,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": std,
        "percentile_95": percentile_95,
        "percentile_99": percentile_99,
    },
    OUTPUT_PATH
)

print("\nSaved calibration file:")
print(OUTPUT_PATH)

print("=" * 50)