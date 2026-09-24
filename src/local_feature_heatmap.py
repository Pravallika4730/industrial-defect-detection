from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


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

STATS_PATH = (
    MODEL_DIR
    / "local_feature_stats.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "heatmaps"
)


# --------------------------------------------------
# SELECT TEST IMAGE
# --------------------------------------------------

CATEGORY = "broken_large"

test_dir = (
    DATASET_DIR
    / "test"
    / CATEGORY
)

image_paths = sorted(
    test_dir.glob("*.png")
)

if not image_paths:
    raise FileNotFoundError(
        f"No images found in {test_dir}"
    )

image_path = image_paths[0]


# --------------------------------------------------
# GROUND TRUTH MASK
# --------------------------------------------------

mask_dir = (
    DATASET_DIR
    / "ground_truth"
    / CATEGORY
)

mask_path = (
    mask_dir
    / f"{image_path.stem}_mask.png"
)


# --------------------------------------------------
# TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 60)
print("Local Feature Anomaly Heatmap")
print("=" * 60)

print(f"\nDevice: {device}")

print(
    f"\nImage: {image_path.name}"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

from model import ResNet18FeatureExtractor


model = ResNet18FeatureExtractor()

model = model.to(device)

model.eval()


# --------------------------------------------------
# SPATIAL FEATURE EXTRACTOR
# --------------------------------------------------

feature_extractor = torch.nn.Sequential(
    *list(model.features.children())[:-1]
)

feature_extractor = (
    feature_extractor.to(device)
)

feature_extractor.eval()


# --------------------------------------------------
# LOAD LOCAL STATISTICS
# --------------------------------------------------

stats = torch.load(
    STATS_PATH,
    map_location=device,
    weights_only=True
)

normal_mean = stats["mean"].to(device)

normal_std = stats["std"].to(device)


print(
    f"\nNormal mean shape: "
    f"{normal_mean.shape}"
)

print(
    f"Normal std shape: "
    f"{normal_std.shape}"
)


# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

original_image = Image.open(
    image_path
).convert("RGB")


image_tensor = transform(
    original_image
)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)


# --------------------------------------------------
# EXTRACT SPATIAL FEATURES
# --------------------------------------------------

with torch.no_grad():

    feature_map = feature_extractor(
        image_tensor
    )


print(
    f"\nFeature map shape: "
    f"{feature_map.shape}"
)


# --------------------------------------------------
# CALCULATE LOCAL ANOMALY SCORE
# --------------------------------------------------

# feature_map:
#
# [1, 512, 8, 8]
#
# normal_mean:
#
# [512, 8, 8]
#
# normal_std:
#
# [512, 8, 8]

difference = (
    feature_map
    - normal_mean.unsqueeze(0)
)


# Standardized difference.

standardized_difference = (
    difference
    / normal_std.unsqueeze(0)
)


# Aggregate across the 512 feature channels.

anomaly_map = torch.norm(
    standardized_difference,
    dim=1
)


# Result:
#
# [1, 8, 8]


anomaly_map = anomaly_map.squeeze(0)


# --------------------------------------------------
# NORMALIZE ANOMALY MAP
# --------------------------------------------------

anomaly_map = anomaly_map.cpu()

minimum = anomaly_map.min()

maximum = anomaly_map.max()

anomaly_map = (
    anomaly_map - minimum
) / (
    maximum - minimum + 1e-8
)


# --------------------------------------------------
# UPSCALE TO IMAGE SIZE
# --------------------------------------------------

anomaly_map = (
    anomaly_map
    .unsqueeze(0)
    .unsqueeze(0)
)


anomaly_map = F.interpolate(
    anomaly_map,
    size=(256, 256),
    mode="bilinear",
    align_corners=False
)


anomaly_map = (
    anomaly_map
    .squeeze()
    .numpy()
)


# --------------------------------------------------
# LOAD GROUND TRUTH
# --------------------------------------------------

ground_truth = None

if mask_path.exists():

    ground_truth = Image.open(
        mask_path
    ).convert("L")

    ground_truth = ground_truth.resize(
        (256, 256)
    )

    ground_truth = (
        torch.from_numpy(
            __import__("numpy").array(
                ground_truth
            )
        ).numpy()
    )

    ground_truth = (
        ground_truth > 0
    )

    print(
        "\nGround-truth mask found."
    )

else:

    print(
        "\nGround-truth mask not found."
    )


# --------------------------------------------------
# CREATE OUTPUT DIRECTORY
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# CREATE VISUALIZATION
# --------------------------------------------------

if ground_truth is not None:

    plt.figure(
        figsize=(15, 5)
    )


    # Original image

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        original_image
    )

    plt.title(
        "Original Image"
    )

    plt.axis("off")


    # Ground truth

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        ground_truth,
        cmap="gray"
    )

    plt.title(
        "Ground Truth"
    )

    plt.axis("off")


    # New anomaly heatmap

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        original_image
    )

    plt.imshow(
        anomaly_map,
        cmap="jet",
        alpha=0.55
    )

    plt.title(
        "Local Feature Anomaly Map"
    )

    plt.axis("off")

else:

    plt.figure(
        figsize=(10, 5)
    )

    plt.subplot(
        1,
        2,
        1
    )

    plt.imshow(
        original_image
    )

    plt.title(
        "Original Image"
    )

    plt.axis("off")


    plt.subplot(
        1,
        2,
        2
    )

    plt.imshow(
        original_image
    )

    plt.imshow(
        anomaly_map,
        cmap="jet",
        alpha=0.55
    )

    plt.title(
        "Local Feature Anomaly Map"
    )

    plt.axis("off")


# --------------------------------------------------
# SAVE RESULT
# --------------------------------------------------

output_path = (
    OUTPUT_DIR
    / f"local_{CATEGORY}_{image_path.stem}.png"
)


plt.tight_layout()

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)


print("\n" + "=" * 60)
print("LOCAL FEATURE HEATMAP CREATED")
print("=" * 60)

print(
    f"\nSaved to:"
)

print(output_path)

print(
    "\nImproved heatmap generated successfully! ✅"
)


plt.show()