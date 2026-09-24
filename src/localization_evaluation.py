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
print("MVTec Bottle - Localization Evaluation")
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
# LOAD NORMAL CENTER
# --------------------------------------------------

normal_center = torch.load(
    CENTER_PATH,
    map_location=device,
    weights_only=True
)

normal_center = normal_center.to(device)


# --------------------------------------------------
# CREATE SPATIAL FEATURE EXTRACTOR
# --------------------------------------------------

feature_extractor = torch.nn.Sequential(
    *list(model.features.children())[:-1]
)

feature_extractor = feature_extractor.to(device)

feature_extractor.eval()


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

    # [1, 512, H, W]
    feature_map = feature_map.squeeze(0)

    # [H, W, 512]
    feature_map = feature_map.permute(
        1,
        2,
        0
    )

    # Compare every spatial feature
    # with the normal feature center.

    difference = (
        feature_map - normal_center
    )

    anomaly_map = torch.norm(
        difference,
        dim=2
    )

    # Normalize
    anomaly_map = anomaly_map.cpu()

    minimum = anomaly_map.min()

    maximum = anomaly_map.max()

    anomaly_map = (
        anomaly_map - minimum
    ) / (
        maximum - minimum + 1e-8
    )

    # Resize to image size

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

    anomaly_map = anomaly_map.squeeze()

    return anomaly_map.numpy()


# --------------------------------------------------
# IOU FUNCTION
# --------------------------------------------------

def calculate_iou(
    predicted_mask,
    ground_truth_mask
):

    intersection = (
        predicted_mask
        & ground_truth_mask
    ).sum()

    union = (
        predicted_mask
        | ground_truth_mask
    ).sum()

    if union == 0:
        return 1.0

    return intersection / union


# --------------------------------------------------
# SELECT IMAGES
# --------------------------------------------------

categories = [
    "broken_large",
    "broken_small",
    "contamination"
]


# --------------------------------------------------
# OUTPUT DIRECTORY
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# EVALUATE
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

    # Evaluate first 5 images from
    # each category for a quick baseline.

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
        # ANOMALY MAP
        # ------------------------------------------

        anomaly_map = get_anomaly_map(
            image_path
        )


        # ------------------------------------------
        # GROUND TRUTH
        # ------------------------------------------

        ground_truth = Image.open(
            mask_path
        ).convert("L")

        ground_truth = ground_truth.resize(
            (256, 256)
        )

        ground_truth = torch.tensor(
            list(ground_truth.getdata())
        ).reshape(
            256,
            256
        ).numpy()

        ground_truth_mask = (
            ground_truth > 0
        )


        # ------------------------------------------
        # PREDICTED MASK
        # ------------------------------------------

        # Use the top 10% anomaly scores
        # as the predicted defect region.

        threshold = torch.tensor(
            anomaly_map
        ).quantile(
            0.90
        ).item()

        predicted_mask = (
            anomaly_map >= threshold
        )


        # ------------------------------------------
        # IOU
        # ------------------------------------------

        iou = calculate_iou(
            predicted_mask,
            ground_truth_mask
        )

        category_ious.append(
            iou
        )

        all_ious.append(
            iou
        )

        print(
            f"{image_path.name:<12} "
            f"IoU: {iou:.4f}"
        )


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
print("LOCALIZATION EVALUATION RESULTS")
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

else:

    print(
        "\nNo valid localization results."
    )


print(
    "\nLocalization evaluation completed! ✅"
)