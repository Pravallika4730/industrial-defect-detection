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

CENTER_PATH = MODEL_DIR / "normal_center.pt"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "heatmaps"
)


# --------------------------------------------------
# SELECT TEST IMAGE
# --------------------------------------------------

CATEGORY = "broken_large"

test_dir = DATASET_DIR / "test" / CATEGORY

image_paths = sorted(test_dir.glob("*.png"))

if len(image_paths) == 0:
    raise FileNotFoundError(
        f"No images found in {test_dir}"
    )

image_path = image_paths[0]

print("=" * 60)
print("MVTec Bottle - Defect Localization")
print("=" * 60)

print(f"\nSelected image:")
print(image_path)


# --------------------------------------------------
# FIND GROUND TRUTH MASK
# --------------------------------------------------

mask_dir = (
    DATASET_DIR
    / "ground_truth"
    / CATEGORY
)

mask_path = mask_dir / f"{image_path.stem}_mask.png"

if mask_path.exists():
    ground_truth = Image.open(mask_path).convert("L")
    print(f"\nGround-truth mask found:")
    print(mask_path)
else:
    ground_truth = None
    print("\nGround-truth mask not found.")


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

print(f"\nDevice: {device}")


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

from model import ResNet18FeatureExtractor


model = ResNet18FeatureExtractor()

model = model.to(device)

model.eval()


# --------------------------------------------------
# LOAD NORMAL FEATURE CENTER
# --------------------------------------------------

normal_center = torch.load(
    CENTER_PATH,
    map_location=device,
    weights_only=True
)

normal_center = normal_center.to(device)

print(
    f"Normal center shape: "
    f"{normal_center.shape}"
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

    # Remove the final average pooling layer.
    #
    # ResNet18 normally produces:
    # [batch, 512, 8, 8]
    #
    # for a 256x256 image.

    feature_extractor = torch.nn.Sequential(
        *list(model.features.children())[:-1]
    )

    feature_map = feature_extractor(
        image_tensor
    )


print(
    f"\nFeature map shape: "
    f"{feature_map.shape}"
)


# --------------------------------------------------
# CREATE ANOMALY MAP
# --------------------------------------------------

# Feature map:
#
# [1, 512, H, W]
#
# Convert it to:
#
# [H, W, 512]
#
# so every spatial location has
# a 512-dimensional feature vector.

feature_map = feature_map.squeeze(0)

feature_map = feature_map.permute(
    1,
    2,
    0
)


# Compare every spatial feature vector
# with the normal feature center.

difference = (
    feature_map
    - normal_center
)


anomaly_map = torch.norm(
    difference,
    dim=2
)


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
# RESIZE HEATMAP
# --------------------------------------------------

anomaly_map = anomaly_map.unsqueeze(0).unsqueeze(0)

anomaly_map = F.interpolate(
    anomaly_map,
    size=(256, 256),
    mode="bilinear",
    align_corners=False
)

anomaly_map = anomaly_map.squeeze().numpy()


# --------------------------------------------------
# PREPARE GROUND TRUTH
# --------------------------------------------------

if ground_truth is not None:

    ground_truth = ground_truth.resize(
        (256, 256)
    )

    ground_truth = (
        torch.tensor(
            list(ground_truth.getdata())
        )
        .reshape(256, 256)
        .numpy()
    )

    ground_truth = (
        ground_truth > 0
    )


# --------------------------------------------------
# CREATE OUTPUT DIRECTORY
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

if ground_truth is not None:

    plt.figure(
        figsize=(15, 5)
    )

    # -------------------------------
    # ORIGINAL
    # -------------------------------

    plt.subplot(1, 3, 1)

    plt.imshow(
        original_image
    )

    plt.title(
        "Original Image"
    )

    plt.axis("off")


    # -------------------------------
    # GROUND TRUTH
    # -------------------------------

    plt.subplot(1, 3, 2)

    plt.imshow(
        ground_truth,
        cmap="gray"
    )

    plt.title(
        "Ground Truth Defect Mask"
    )

    plt.axis("off")


    # -------------------------------
    # ANOMALY HEATMAP
    # -------------------------------

    plt.subplot(1, 3, 3)

    plt.imshow(
        original_image
    )

    plt.imshow(
        anomaly_map,
        cmap="jet",
        alpha=0.55
    )

    plt.title(
        "Anomaly Heatmap"
    )

    plt.axis("off")


else:

    plt.figure(
        figsize=(10, 5)
    )

    plt.subplot(1, 2, 1)

    plt.imshow(
        original_image
    )

    plt.title(
        "Original Image"
    )

    plt.axis("off")


    plt.subplot(1, 2, 2)

    plt.imshow(
        original_image
    )

    plt.imshow(
        anomaly_map,
        cmap="jet",
        alpha=0.55
    )

    plt.title(
        "Anomaly Heatmap"
    )

    plt.axis("off")


# --------------------------------------------------
# SAVE RESULT
# --------------------------------------------------

output_path = (
    OUTPUT_DIR
    / f"{CATEGORY}_{image_path.stem}_heatmap.png"
)


plt.tight_layout()

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)


print("\n" + "=" * 60)
print("HEATMAP GENERATED")
print("=" * 60)

print(
    f"\nHeatmap saved to:"
)

print(output_path)

print(
    "\nDefect localization completed successfully! ✅"
)


plt.show()