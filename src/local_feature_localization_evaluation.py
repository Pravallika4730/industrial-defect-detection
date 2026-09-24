from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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

STATS_PATH = (
    MODEL_DIR
    / "local_feature_stats.pt"
)


# --------------------------------------------------
# IMAGE TRANSFORM
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
print("Local Feature - Localization Evaluation")
print("=" * 60)

print(f"\nDevice: {device}")


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

feature_extractor = feature_extractor.to(device)

feature_extractor.eval()


# --------------------------------------------------
# LOAD NORMAL LOCAL STATISTICS
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
# ANOMALY MAP FUNCTION
# --------------------------------------------------

def get_anomaly_map(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        feature_map = feature_extractor(
            image_tensor
        )

    # feature_map:
    # [1, 512, 8, 8]

    difference = (
        feature_map
        - normal_mean.unsqueeze(0)
    )

    standardized_difference = (
        difference
        / normal_std.unsqueeze(0)
    )

    # Aggregate the 512 channels.

    anomaly_map = torch.norm(
        standardized_difference,
        dim=1
    )

    # [1, 8, 8]

    anomaly_map = anomaly_map.squeeze(0)

    # Normalize.

    anomaly_map = anomaly_map.cpu()

    minimum = anomaly_map.min()

    maximum = anomaly_map.max()

    anomaly_map = (
        anomaly_map - minimum
    ) / (
        maximum - minimum + 1e-8
    )

    # Upscale 8x8 -> 256x256.

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

    return anomaly_map.squeeze().numpy()


# --------------------------------------------------
# IOU FUNCTION
# --------------------------------------------------

def calculate_iou(
    predicted_mask,
    ground_truth_mask
):

    intersection = np.logical_and(
        predicted_mask,
        ground_truth_mask
    ).sum()

    union = np.logical_or(
        predicted_mask,
        ground_truth_mask
    ).sum()

    if union == 0:
        return 1.0

    return intersection / union


# --------------------------------------------------
# TEST CATEGORIES
# --------------------------------------------------

categories = [
    "broken_large",
    "broken_small",
    "contamination"
]


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

all_ious = []


for category in categories:

    test_dir = (
        DATASET_DIR
        / "test"
        / category
    )

    mask_dir = (
        DATASET_DIR
        / "ground_truth"
        / category
    )

    image_paths = sorted(
        test_dir.glob("*.png")
    )

    category_ious = []

    print("\n" + "-" * 60)

    print(
        f"Category: {category}"
    )


    # Same 5 images per category
    # used in the previous experiment.

    for image_path in image_paths[:5]:

        mask_path = (
            mask_dir
            / f"{image_path.stem}_mask.png"
        )


        if not mask_path.exists():

            print(
                f"Mask not found: "
                f"{image_path.name}"
            )

            continue


        # ------------------------------------------
        # GET ANOMALY MAP
        # ------------------------------------------

        anomaly_map = get_anomaly_map(
            image_path
        )


        # ------------------------------------------
        # LOAD GROUND TRUTH
        # ------------------------------------------

        ground_truth = Image.open(
            mask_path
        ).convert("L")

        ground_truth = ground_truth.resize(
            (256, 256)
        )

        ground_truth = np.array(
            ground_truth
        )

        ground_truth_mask = (
            ground_truth > 0
        )


        # ------------------------------------------
        # PREDICTED DEFECT REGION
        # ------------------------------------------

        # Use the top 10% anomaly scores,
        # exactly as in the previous method.

        threshold = np.quantile(
            anomaly_map,
            0.90
        )

        predicted_mask = (
            anomaly_map >= threshold
        )


        # ------------------------------------------
        # CALCULATE IOU
        # ------------------------------------------

        iou = calculate_iou(
            predicted_mask,
            ground_truth_mask
        )

        category_ious.append(iou)

        all_ious.append(iou)


        print(
            f"{image_path.name:<12} "
            f"IoU: {iou:.4f}"
        )


    # ------------------------------------------
    # CATEGORY MEAN
    # ------------------------------------------

    if category_ious:

        average_iou = (
            sum(category_ious)
            / len(category_ious)
        )

        print(
            f"\nAverage IoU: "
            f"{average_iou:.4f}"
        )


# --------------------------------------------------
# OVERALL RESULT
# --------------------------------------------------

print("\n" + "=" * 60)
print("LOCAL FEATURE LOCALIZATION RESULTS")
print("=" * 60)


if all_ious:

    overall_iou = (
        sum(all_ious)
        / len(all_ious)
    )

    print(
        f"\nImages evaluated: "
        f"{len(all_ious)}"
    )

    print(
        f"Mean IoU: "
        f"{overall_iou:.4f}"
    )

    print(
        f"\nPrevious baseline IoU: "
        f"0.0997"
    )

    improvement = (
        overall_iou - 0.0997
    )

    print(
        f"Absolute change: "
        f"{improvement:+.4f}"
    )

else:

    print(
        "\nNo valid localization results."
    )


print(
    "\nLocal feature localization "
    "evaluation completed! ✅"
)