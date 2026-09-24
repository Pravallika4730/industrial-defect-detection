from pathlib import Path
import csv

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

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
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# ============================================================
# LOAD RESNET18 FEATURE EXTRACTOR
# ============================================================

model = ResNet18FeatureExtractor().to(device)
model.eval()


# ============================================================
# LOAD NORMAL CENTER
# ============================================================
# normal_center.pt contains the feature-center tensor directly.
# It is NOT a dictionary.

normal_center = torch.load(
    MODEL_DIR / "normal_center.pt",
    map_location=device,
    weights_only=True
)

normal_center = normal_center.to(device)

print(f"Normal center shape: {normal_center.shape}")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ============================================================
# VALIDATED THRESHOLD
# ============================================================
# This threshold was obtained from the original validated model.

threshold = 4.0173

print(f"Validated threshold: {threshold:.4f}")


# ============================================================
# ANOMALY SCORE FUNCTION
# ============================================================

def calculate_score(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        features = model(image_tensor)

        distance = torch.norm(
            features - normal_center,
            dim=1
        )

    return distance.item()


# ============================================================
# TEST CATEGORIES
# ============================================================

categories = [
    "good",
    "broken_large",
    "broken_small",
    "contamination",
]


# ============================================================
# CONFUSION MATRIX COUNTERS
# ============================================================

TP = 0
TN = 0
FP = 0
FN = 0


# ============================================================
# STORAGE FOR PREDICTIONS
# ============================================================

rows = []

category_stats = {}


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 65)
print("FINAL EVALUATION - VALIDATED GLOBAL MODEL")
print("=" * 65)


for category in categories:

    category_dir = DATASET_DIR / "test" / category

    image_paths = sorted(
        category_dir.glob("*.png")
    )

    correct = 0

    # good = normal
    # all other categories = defective

    actual_defect = category != "good"

    for image_path in image_paths:

        score = calculate_score(image_path)

        predicted_defect = score > threshold


        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        if actual_defect and predicted_defect:

            TP += 1

        elif not actual_defect and not predicted_defect:

            TN += 1

        elif not actual_defect and predicted_defect:

            FP += 1

        elif actual_defect and not predicted_defect:

            FN += 1


        # ----------------------------------------------------
        # CATEGORY ACCURACY
        # ----------------------------------------------------

        if predicted_defect == actual_defect:

            correct += 1


        # ----------------------------------------------------
        # STORE PREDICTION
        # ----------------------------------------------------

        rows.append({
            "image": image_path.name,
            "category": category,
            "actual": (
                "DEFECT"
                if actual_defect
                else "GOOD"
            ),
            "predicted": (
                "DEFECT"
                if predicted_defect
                else "GOOD"
            ),
            "score": round(score, 4),
            "correct": predicted_defect == actual_defect
        })


    # --------------------------------------------------------
    # CATEGORY RESULTS
    # --------------------------------------------------------

    category_accuracy = (
        correct / len(image_paths)
    )

    category_stats[category] = {
        "correct": correct,
        "total": len(image_paths),
        "accuracy": category_accuracy
    }

    print(
        f"{category:18s} "
        f"{correct}/{len(image_paths)} "
        f"({category_accuracy * 100:.2f}%)"
    )


# ============================================================
# METRICS
# ============================================================

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


# ============================================================
# CONFUSION MATRIX RESULTS
# ============================================================

print("\n" + "=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

print(f"True Positive  (TP): {TP}")
print(f"True Negative  (TN): {TN}")
print(f"False Positive (FP): {FP}")
print(f"False Negative (FN): {FN}")


# ============================================================
# FINAL METRICS
# ============================================================

print("\n" + "=" * 65)
print("FINAL METRICS")
print("=" * 65)

print(
    f"Accuracy  : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision : {precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall    : {recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"F1 Score  : {f1:.4f} "
    f"({f1 * 100:.2f}%)"
)


# ============================================================
# SAVE PREDICTIONS CSV
# ============================================================

csv_path = (
    RESULTS_DIR
    / "final_evaluation_predictions.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image",
            "category",
            "actual",
            "predicted",
            "score",
            "correct"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


print("\nPrediction CSV saved:")
print(csv_path)


# ============================================================
# METRICS GRAPH
# ============================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

metric_values = [
    accuracy * 100,
    precision * 100,
    recall * 100,
    f1 * 100
]

plt.figure(figsize=(9, 6))

bars = plt.bar(
    metric_names,
    metric_values
)

plt.ylim(0, 100)

plt.ylabel("Percentage (%)")

plt.title(
    "Industrial Surface Defect Detection - Final Metrics"
)

plt.grid(
    axis="y",
    alpha=0.3
)

for bar, value in zip(
    bars,
    metric_values
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()

metrics_graph = (
    RESULTS_DIR
    / "final_metrics.png"
)

plt.savefig(
    metrics_graph,
    dpi=300
)

plt.close()

print("\nMetrics graph saved:")
print(metrics_graph)


# ============================================================
# CONFUSION MATRIX GRAPH
# ============================================================

matrix = [
    [TN, FP],
    [FN, TP]
]

plt.figure(figsize=(7, 6))

plt.imshow(matrix)

plt.title(
    "Confusion Matrix"
)

plt.xticks(
    [0, 1],
    [
        "Predicted GOOD",
        "Predicted DEFECT"
    ]
)

plt.yticks(
    [0, 1],
    [
        "Actual GOOD",
        "Actual DEFECT"
    ]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            matrix[i][j],
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold"
        )

plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.tight_layout()

confusion_graph = (
    RESULTS_DIR
    / "confusion_matrix.png"
)

plt.savefig(
    confusion_graph,
    dpi=300
)

plt.close()

print("\nConfusion matrix saved:")
print(confusion_graph)


# ============================================================
# FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

print("Model       : Pretrained ResNet18")
print("Dataset     : MVTec AD - Bottle")
print("Test Images : 83")
print("Threshold   : 4.0173")

print("\nClassification Performance:")

print(
    f"Accuracy    : {accuracy * 100:.2f}%"
)

print(
    f"Precision   : {precision * 100:.2f}%"
)

print(
    f"Recall      : {recall * 100:.2f}%"
)

print(
    f"F1 Score    : {f1 * 100:.2f}%"
)

print("\nEvaluation completed successfully!")