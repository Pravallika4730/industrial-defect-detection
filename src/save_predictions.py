from pathlib import Path
import csv

import torch
from torchvision import transforms
from PIL import Image

from model import ResNet18FeatureExtractor


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "mvtec_anomaly_detection"
    / "bottle"
)

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

CENTER_PATH = MODEL_DIR / "normal_center.pt"
OUTPUT_PATH = RESULTS_DIR / "bottle_predictions.csv"


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("MVTec Bottle - Saving Predictions")
print("=" * 60)

print(f"\nDevice: {device}")


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = ResNet18FeatureExtractor()
model = model.to(device)
model.eval()


# --------------------------------------------------
# LOAD NORMAL CENTER
# --------------------------------------------------

normal_center = torch.load(
    CENTER_PATH,
    map_location=device,
    weights_only=True
)

normal_center = normal_center.to(device)


# --------------------------------------------------
# ANOMALY SCORE FUNCTION
# --------------------------------------------------

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


# --------------------------------------------------
# TEST CATEGORIES
# --------------------------------------------------

test_categories = {
    "good": 0,
    "broken_large": 1,
    "broken_small": 1,
    "contamination": 1,
}


# --------------------------------------------------
# CALCULATE THRESHOLD
# --------------------------------------------------

train_dir = DATASET_DIR / "train" / "good"

training_scores = []

for image_path in sorted(train_dir.glob("*.png")):

    score = get_anomaly_score(image_path)

    training_scores.append(score)


threshold = max(training_scores)

print(
    f"\nDetection threshold: {threshold:.4f}"
)


# --------------------------------------------------
# PROCESS TEST IMAGES
# --------------------------------------------------

results = []

for category, true_label in test_categories.items():

    category_dir = DATASET_DIR / "test" / category

    images = sorted(category_dir.glob("*.png"))

    print(
        f"\nProcessing {category}: "
        f"{len(images)} images"
    )

    for image_path in images:

        score = get_anomaly_score(image_path)

        predicted_label = (
            1 if score > threshold else 0
        )

        predicted_class = (
            "defect"
            if predicted_label == 1
            else "good"
        )

        actual_class = (
            "defect"
            if true_label == 1
            else "good"
        )

        results.append([
            category,
            image_path.name,
            actual_class,
            predicted_class,
            round(score, 4),
            threshold,
        ])


# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Category",
        "Image",
        "Actual",
        "Predicted",
        "Anomaly_Score",
        "Threshold",
    ])

    writer.writerows(results)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

correct = 0

for row in results:

    actual = row[2]
    predicted = row[3]

    if actual == predicted:
        correct += 1


accuracy = correct / len(results)


print("\n" + "=" * 60)
print("PREDICTION FILE CREATED")
print("=" * 60)

print(f"\nTotal predictions: {len(results)}")

print(
    f"Correct predictions: {correct}"
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print("\nCSV saved to:")

print(OUTPUT_PATH)

print("\nPrediction export completed successfully! ✅")