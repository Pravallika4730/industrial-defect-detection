from pathlib import Path
import csv

import matplotlib.pyplot as plt


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "bottle_predictions.csv"
)

GRAPH_DIR = (
    PROJECT_ROOT
    / "results"
    / "graphs"
)

OUTPUT_PATH = (
    GRAPH_DIR
    / "bottle_category_analysis.png"
)


# --------------------------------------------------
# READ CSV
# --------------------------------------------------

category_data = {}

with open(
    CSV_PATH,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        category = row["Category"]
        actual = row["Actual"]
        predicted = row["Predicted"]

        score = float(row["Anomaly_Score"])

        if category not in category_data:
            category_data[category] = {
                "scores": [],
                "correct": 0,
                "total": 0,
                "actual": actual
            }

        category_data[category]["scores"].append(score)

        category_data[category]["total"] += 1

        if actual == predicted:
            category_data[category]["correct"] += 1


# --------------------------------------------------
# PRINT CATEGORY RESULTS
# --------------------------------------------------

print("=" * 60)
print("MVTec Bottle - Per Category Analysis")
print("=" * 60)

print()

category_names = []
accuracies = []

for category, data in category_data.items():

    accuracy = (
        data["correct"] / data["total"]
    )

    category_names.append(category)
    accuracies.append(accuracy * 100)

    average_score = (
        sum(data["scores"])
        / len(data["scores"])
    )

    minimum_score = min(data["scores"])
    maximum_score = max(data["scores"])

    print(
        f"{category:<18} "
        f"Accuracy: {accuracy * 100:6.2f}%   "
        f"Avg Score: {average_score:6.2f}   "
        f"Min: {minimum_score:6.2f}   "
        f"Max: {maximum_score:6.2f}"
    )


# --------------------------------------------------
# CREATE GRAPH DIRECTORY
# --------------------------------------------------

GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# CREATE BAR CHART
# --------------------------------------------------

plt.figure(figsize=(10, 6))

bars = plt.bar(
    category_names,
    accuracies
)


plt.xlabel("Test Category")

plt.ylabel("Accuracy (%)")

plt.title(
    "MVTec Bottle - Accuracy by Test Category"
)

plt.ylim(0, 100)

plt.grid(
    axis="y",
    alpha=0.3
)


# --------------------------------------------------
# ADD VALUES ABOVE BARS
# --------------------------------------------------

for bar, accuracy in zip(
    bars,
    accuracies
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.1f}%",
        ha="center"
    )


plt.tight_layout()


# --------------------------------------------------
# SAVE GRAPH
# --------------------------------------------------

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)


print("\n" + "=" * 60)

print("CATEGORY ANALYSIS COMPLETED")
print("=" * 60)

print("\nGraph saved to:")

print(OUTPUT_PATH)

print("\nDone! ✅")


# --------------------------------------------------
# SHOW GRAPH
# --------------------------------------------------

plt.show()