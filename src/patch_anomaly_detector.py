from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image


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

OUTPUT_PATH = (
    MODEL_DIR
    / "local_feature_stats.pt"
)


# --------------------------------------------------
# IMAGE TRANSFORMATION
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
print("Local Feature Anomaly Detector")
print("=" * 60)

print(f"\nDevice: {device}")


# --------------------------------------------------
# LOAD RESNET18
# --------------------------------------------------

from model import ResNet18FeatureExtractor


model = ResNet18FeatureExtractor()

model = model.to(device)

model.eval()


# --------------------------------------------------
# CREATE SPATIAL FEATURE EXTRACTOR
# --------------------------------------------------

# model.features contains:
#
# Conv
# BatchNorm
# ReLU
# MaxPool
# Layer1
# Layer2
# Layer3
# Layer4
# AveragePool
#
# We remove AveragePool so that we keep
# the spatial feature map.

feature_extractor = torch.nn.Sequential(
    *list(model.features.children())[:-1]
)

feature_extractor = feature_extractor.to(device)

feature_extractor.eval()


# --------------------------------------------------
# TRAINING DATA
# --------------------------------------------------

train_dir = (
    DATASET_DIR
    / "train"
    / "good"
)

image_paths = sorted(
    train_dir.glob("*.png")
)

print(
    f"\nNormal training images: "
    f"{len(image_paths)}"
)


# --------------------------------------------------
# EXTRACT SPATIAL FEATURES
# --------------------------------------------------

feature_maps = []


for index, image_path in enumerate(
    image_paths
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        device
    )

    with torch.no_grad():

        features = feature_extractor(
            image_tensor
        )

    # Expected:
    #
    # [1, 512, 8, 8]

    features = features.squeeze(0)

    feature_maps.append(
        features.cpu()
    )

    if (index + 1) % 25 == 0:

        print(
            f"Processed "
            f"{index + 1}/"
            f"{len(image_paths)} images"
        )


# --------------------------------------------------
# STACK FEATURE MAPS
# --------------------------------------------------

feature_maps = torch.stack(
    feature_maps
)


print(
    "\nFeature tensor shape:"
)

print(
    feature_maps.shape
)


# --------------------------------------------------
# CALCULATE LOCAL NORMAL STATISTICS
# --------------------------------------------------

# Shape:
#
# [209, 512, 8, 8]
#
# Calculate statistics across training images.

normal_mean = feature_maps.mean(
    dim=0
)

normal_std = feature_maps.std(
    dim=0
)


# Prevent division by zero.

normal_std = torch.clamp(
    normal_std,
    min=1e-6
)


print(
    "\nNormal mean shape:"
)

print(
    normal_mean.shape
)


print(
    "\nNormal standard deviation shape:"
)

print(
    normal_std.shape
)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


torch.save(
    {
        "mean": normal_mean,
        "std": normal_std,
    },
    OUTPUT_PATH
)


# --------------------------------------------------
# VERIFY SAVED MODEL
# --------------------------------------------------

saved_model = torch.load(
    OUTPUT_PATH,
    map_location="cpu",
    weights_only=True
)


print(
    "\nSaved model contains:"
)

print(
    saved_model.keys()
)


print("\n" + "=" * 60)

print(
    "LOCAL FEATURE MODEL CREATED"
)

print("=" * 60)

print(
    "\nModel saved to:"
)

print(OUTPUT_PATH)

print(
    "\nLocal feature anomaly detector "
    "created successfully! ✅"
)
