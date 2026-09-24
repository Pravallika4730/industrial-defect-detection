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
    / "bottle_anomaly_scores.png"
)


# --------------------------------------------------
# READ CSV
# --------------------------------------------------

good_scores = []
defect_scores = []

threshold = None


with open(
    CSV_PATH,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        score = float(row["Anomaly_Score"])

        threshold = float(row["Threshold"])

        if row["Actual"] == "good":
            good_scores.append(score)

        else:
            defect_scores.append(score)


# --------------------------------------------------
# CREATE GRAPH DIRECTORY
# --------------------------------------------------

GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# CREATE PLOT
# --------------------------------------------------

plt.figure(figsize=(10, 6))


plt.hist(
    good_scores,
    bins=10,
    alpha=0.7,
    label="Good / Normal"
)


plt.hist(
    defect_scores,
    bins=15,
    alpha=0.7,
    label="Defective"
)


plt.axvline(
    threshold,
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {threshold:.2f}"
)


plt.xlabel("Anomaly Score")

plt.ylabel("Number of Images")

plt.title(
    "MVTec Bottle - Anomaly Score Distribution"
)

plt.legend()

plt.grid(
    alpha=0.3
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


print("=" * 60)
print("ANOMALY SCORE VISUALIZATION")
print("=" * 60)

print(f"\nGood images    : {len(good_scores)}")
print(f"Defective images: {len(defect_scores)}")

print(
    f"\nThreshold      : {threshold:.4f}"
)

print("\nGraph saved to:")

print(OUTPUT_PATH)

print(
    "\nVisualization completed successfully! ✅"
)


# --------------------------------------------------
# SHOW GRAPH
# --------------------------------------------------

plt.show()