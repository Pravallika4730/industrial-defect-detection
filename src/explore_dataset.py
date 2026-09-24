from pathlib import Path
import cv2
import matplotlib.pyplot as plt


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

TRAIN_DIR = DATASET_DIR / "train" / "good"
TEST_DIR = DATASET_DIR / "test"


# ==========================================
# 2. COUNT IMAGES
# ==========================================

train_images = list(TRAIN_DIR.glob("*.png"))

print("=" * 60)
print("MVTec AD - Bottle Dataset Exploration")
print("=" * 60)

print(f"\nTraining images (good): {len(train_images)}")

print("\nTest categories:")

test_categories = sorted(
    [folder for folder in TEST_DIR.iterdir() if folder.is_dir()]
)

for category in test_categories:
    images = list(category.glob("*.png"))
    print(f"  {category.name:20s}: {len(images)} images")


# ==========================================
# 3. SELECT SAMPLE IMAGES
# ==========================================

normal_image_path = train_images[0]

good_test_images = list((TEST_DIR / "good").glob("*.png"))
broken_large_images = list((TEST_DIR / "broken_large").glob("*.png"))

good_test_image_path = good_test_images[0]
defect_image_path = broken_large_images[0]


# ==========================================
# 4. LOAD IMAGES USING OPENCV
# ==========================================

normal_image = cv2.imread(str(normal_image_path))
good_test_image = cv2.imread(str(good_test_image_path))
defect_image = cv2.imread(str(defect_image_path))


# Convert BGR → RGB for Matplotlib

normal_image = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
good_test_image = cv2.cvtColor(good_test_image, cv2.COLOR_BGR2RGB)
defect_image = cv2.cvtColor(defect_image, cv2.COLOR_BGR2RGB)


# ==========================================
# 5. DISPLAY IMAGES
# ==========================================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(normal_image)
plt.title("Training - Good")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(good_test_image)
plt.title("Test - Good")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(defect_image)
plt.title("Test - Broken Large")
plt.axis("off")

plt.tight_layout()


# ==========================================
# 6. SAVE VISUALIZATION
# ==========================================

RESULTS_DIR = PROJECT_ROOT / "results" / "graphs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

output_path = RESULTS_DIR / "bottle_dataset_samples.png"

plt.savefig(output_path, dpi=150, bbox_inches="tight")

print("\nVisualization saved to:")
print(output_path)

plt.show()