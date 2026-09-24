from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from model import ResNet18FeatureExtractor


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "mvtec_anomaly_detection"
    / "bottle"
)

MODEL_DIR = PROJECT_ROOT / "models"

CENTER_PATH = MODEL_DIR / "normal_center.pt"


# ==========================================
# 2. IMAGE TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ==========================================
# 3. DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 4. LOAD MODEL
# ==========================================

model = ResNet18FeatureExtractor()

model = model.to(device)

model.eval()


# ==========================================
# 5. LOAD NORMAL CENTER
# ==========================================

normal_center = torch.load(
    CENTER_PATH,
    map_location=device,
    weights_only=True
)

normal_center = normal_center.to(device)


# ==========================================
# 6. PREDICT ANOMALY SCORE
# ==========================================

def get_anomaly_score(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        features = model(image_tensor)

        score = torch.norm(
            features - normal_center,
            dim=1
        )

    return score.item()


# ==========================================
# 7. COLLECT TEST IMAGES
# ==========================================

test_categories = {
    "good": 0,
    "broken_large": 1,
    "broken_small": 1,
    "contamination": 1,
}


results = []


print("=" * 60)
print("MVTec Bottle - Full Test Evaluation")
print("=" * 60)

print(f"\nDevice: {device}")


# ==========================================
# 8. EVALUATE ALL IMAGES
# ==========================================

for category, true_label in test_categories.items():

    category_dir = DATASET_DIR / "test" / category

    images = sorted(category_dir.glob("*.png"))

    print(
        f"\nProcessing {category}: "
        f"{len(images)} images"
    )

    for image_path in images:

        score = get_anomaly_score(image_path)

        results.append({
            "category": category,
            "image": image_path.name,
            "true_label": true_label,
            "score": score
        })


# ==========================================
# 9. TRAINING DISTANCES
# ==========================================

train_dir = DATASET_DIR / "train" / "good"

training_scores = []

for image_path in sorted(train_dir.glob("*.png")):

    score = get_anomaly_score(image_path)

    training_scores.append(score)


training_scores = torch.tensor(training_scores)


# ==========================================
# 10. CHOOSE BASELINE THRESHOLD
# ==========================================

# We use the maximum distance observed
# among normal training images.

threshold = training_scores.max().item()


print("\n" + "=" * 60)
print("BASELINE THRESHOLD")
print("=" * 60)

print(
    f"\nMaximum normal training score: "
    f"{threshold:.4f}"
)


# ==========================================
# 11. CLASSIFY TEST IMAGES
# ==========================================

true_labels = []
predicted_labels = []
scores = []

for result in results:

    true_label = result["true_label"]

    score = result["score"]

    predicted_label = 1 if score > threshold else 0

    true_labels.append(true_label)

    predicted_labels.append(predicted_label)

    scores.append(score)


# ==========================================
# 12. CONFUSION MATRIX
# ==========================================

TP = 0
TN = 0
FP = 0
FN = 0

for true, predicted in zip(
    true_labels,
    predicted_labels
):

    if true == 1 and predicted == 1:
        TP += 1

    elif true == 0 and predicted == 0:
        TN += 1

    elif true == 0 and predicted == 1:
        FP += 1

    elif true == 1 and predicted == 0:
        FN += 1


# ==========================================
# 13. METRICS
# ==========================================

total = TP + TN + FP + FN

accuracy = (
    (TP + TN) / total
    if total > 0
    else 0
)

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


# ==========================================
# 14. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(f"\nTotal test images: {total}")

print("\nConfusion Matrix:")

print(f"True Negatives : {TN}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"True Positives : {TP}")

print("\nMetrics:")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


print("\nEvaluation completed successfully! ✅")