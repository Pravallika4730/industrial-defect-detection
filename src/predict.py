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

NORMAL_CENTER_PATH = MODEL_DIR / "normal_center.pt"


# ==========================================
# 2. DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 3. IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ==========================================
# 4. LOAD RESNET18
# ==========================================

model = ResNet18FeatureExtractor()

model = model.to(device)

model.eval()


# ==========================================
# 5. LOAD NORMAL CENTER
# ==========================================

normal_center = torch.load(
    NORMAL_CENTER_PATH,
    map_location=device,
    weights_only=True
)

normal_center = normal_center.to(device)


# ==========================================
# 6. PREDICTION FUNCTION
# ==========================================

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # Extract features
    with torch.no_grad():

        features = model(image_tensor)

        # Calculate distance from normal center
        distance = torch.norm(
            features - normal_center,
            dim=1
        )

    anomaly_score = distance.item()

    return anomaly_score


# ==========================================
# 7. TEST IMAGES
# ==========================================

test_categories = [
    "good",
    "broken_large",
    "broken_small",
    "contamination"
]


print("=" * 60)
print("MVTec Bottle Anomaly Prediction")
print("=" * 60)

print(f"\nDevice: {device}")


for category in test_categories:

    category_dir = DATASET_DIR / "test" / category

    images = sorted(category_dir.glob("*.png"))

    # Test first image from each category
    image_path = images[0]

    score = predict_image(image_path)

    print("\n----------------------------------------")
    print(f"Category: {category}")
    print(f"Image:    {image_path.name}")
    print(f"Score:    {score:.4f}")


print("\nPrediction test completed successfully! ✅")