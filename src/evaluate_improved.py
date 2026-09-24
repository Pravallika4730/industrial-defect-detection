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
THRESHOLD_PATH = MODEL_DIR / "improved_score_threshold.pt"


# ============================================================
# PREPROCESSING
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

print(f"Device: {device}")


# ============================================================
# LOAD MODEL
# ============================================================

model = ResNet18FeatureExtractor().to(device)
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


# ============================================================
# LOAD CALIBRATED THRESHOLD
# ============================================================

threshold_data = torch.load(
    THRESHOLD_PATH,
    map_location="cpu",
    weights_only=True
)

threshold = threshold_data["threshold"]

print(f"Improved threshold: {threshold:.4f}")


# ============================================================
# ANOMALY SCORE FUNCTION
# ============================================================

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


# ============================================================
# TEST DATA
# ============================================================

test_dir = DATASET_DIR / "test"

categories = [
    "good",
    "broken_large",
    "broken_small",
    "contamination",
]


# ============================================================
# CONFUSION MATRIX
# ============================================================

TP = 0
TN = 0
FP = 0
FN = 0

category_results = {}


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING IMPROVED ANOMALY DETECTOR")
print("=" * 60)

for category in categories:

    category_dir = test_dir / category

    image_paths = sorted(
        category_dir.glob("*.png")
    )

    category_correct = 0

    # Ground truth:
    # good = normal
    # everything else = defect

    actual_defect = category != "good"

    for image_path in image_paths:

        score = calculate_anomaly_score(image_path)

        predicted_defect = score > threshold

        # Confusion matrix
        if actual_defect and predicted_defect:
            TP += 1

        elif not actual_defect and not predicted_defect:
            TN += 1

        elif not actual_defect and predicted_defect:
            FP += 1

        elif actual_defect and not predicted_defect:
            FN += 1

        # Category accuracy
        if predicted_defect == actual_defect:
            category_correct += 1

    total_category = len(image_paths)

    category_accuracy = (
        category_correct / total_category
    )

    category_results[category] = category_accuracy

    print(
        f"{category:18s} "
        f"{category_correct}/{total_category} "
        f"correct "
        f"({category_accuracy * 100:.2f}%)"
    )


# ============================================================
# METRICS
# ============================================================

total = TP + TN + FP + FN

accuracy = (TP + TN) / total

precision = (
    TP / (TP + FP)
    if (TP + FP) > 0
    else 0
)

recall = (
    TP / (TP + FN)
    if (TP + FN) > 0
    else 0
)

f1 = (
    2 * precision * recall
    / (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("IMPROVED MODEL RESULTS")
print("=" * 60)

print(f"Total images : {total}")

print(f"\nTP : {TP}")
print(f"TN : {TN}")
print(f"FP : {FP}")
print(f"FN : {FN}")

print("\nMetrics:")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nPercentages:")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\n" + "=" * 60)
print("COMPARISON WITH CURRENT MODEL")
print("=" * 60)

print("Current model:")
print("Accuracy  : 91.57%")
print("Precision : 98.28%")
print("Recall    : 90.48%")
print("F1 Score  : 94.21%")

print("\nImproved local-score model:")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nEvaluation complete.")