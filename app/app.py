from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st

from PIL import Image, ImageFilter
from torchvision import transforms
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"
MODEL_DIR = PROJECT_ROOT / "models"

sys.path.append(str(SRC_DIR))

from model import ResNet18FeatureExtractor


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Industrial Quality Inspection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #888888;
        margin-bottom: 30px;
    }

    .status-good {
        padding: 22px;
        border-radius: 14px;
        background-color: #e8f5e9;
        border: 2px solid #43a047;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        color: #1b5e20;
    }

    .status-defect {
        padding: 22px;
        border-radius: 14px;
        background-color: #ffebee;
        border: 2px solid #e53935;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        color: #b71c1c;
    }

    .metric-card {
        padding: 18px;
        border-radius: 14px;
        background-color: #f5f7fa;
        border: 1px solid #dfe3e8;
        text-align: center;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #222222;
    }

    .metric-label {
        font-size: 14px;
        color: #555555;
    }

    .technical-card {
        padding: 20px;
        border-radius: 14px;
        background-color: #f5f7fa;
        border: 1px solid #dfe3e8;
        color: #222222;
        min-height: 140px;
    }

    .technical-card-title {
        font-size: 18px;
        font-weight: 700;
        color: #111111;
        margin-bottom: 10px;
    }

    .technical-card-text {
        font-size: 15px;
        color: #444444;
        line-height: 1.6;
    }

    .pipeline-card {
        padding: 20px;
        border-radius: 14px;
        background-color: #f5f7fa;
        border: 1px solid #dfe3e8;
        color: #222222;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🏭 AI Industrial Surface Defect Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning Based Automated Quality Inspection using ResNet18'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ============================================================
# VALIDATED CLASSIFICATION THRESHOLD
# ============================================================

CLASSIFICATION_THRESHOLD = 4.0173


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = ResNet18FeatureExtractor()

    model = model.to(device)

    model.eval()

    return model


# ============================================================
# LOAD NORMAL FEATURE CENTER
# ============================================================

@st.cache_resource
def load_normal_center():

    normal_center = torch.load(
        MODEL_DIR / "normal_center.pt",
        map_location=device,
        weights_only=True
    )

    return normal_center.to(device)


# ============================================================
# LOAD LOCAL FEATURE STATISTICS
# ============================================================

@st.cache_resource
def load_local_statistics():

    stats = torch.load(
        MODEL_DIR / "local_feature_stats.pt",
        map_location=device,
        weights_only=True
    )

    normal_mean = stats["mean"].to(device)
    normal_std = stats["std"].to(device)

    return normal_mean, normal_std


# ============================================================
# CREATE SPATIAL FEATURE EXTRACTOR
# ============================================================

@st.cache_resource
def create_feature_extractor():

    model = load_model()

    feature_extractor = torch.nn.Sequential(
        *list(model.features.children())[:-1]
    )

    feature_extractor = feature_extractor.to(device)

    feature_extractor.eval()

    return feature_extractor


# ============================================================
# GLOBAL ANOMALY SCORE
# ============================================================

def calculate_global_score(
    image,
    model,
    normal_center
):

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        features = model(
            image_tensor
        )

        score = torch.norm(
            features - normal_center,
            dim=1
        )

    return score.item()


# ============================================================
# CREATE FOREGROUND MASK
# ============================================================

def create_foreground_mask(image):

    # Resize image
    resized = image.resize(
        (256, 256)
    )

    # Convert to grayscale
    gray = resized.convert("L")

    # Slight blur to remove tiny noise
    gray = gray.filter(
        ImageFilter.GaussianBlur(radius=2)
    )

    gray_array = np.array(
        gray
    ).astype(np.float32)

    # MVTec bottle images generally have
    # a bright background and darker bottle.
    foreground = gray_array < 245

    # Convert to float
    mask = foreground.astype(
        np.float32
    )

    # Convert to tensor
    mask_tensor = torch.from_numpy(
        mask
    )

    # Smooth mask
    mask_tensor = (
        mask_tensor
        .unsqueeze(0)
        .unsqueeze(0)
    )

    mask_tensor = F.avg_pool2d(
        mask_tensor,
        kernel_size=15,
        stride=1,
        padding=7
    )

    mask_tensor = mask_tensor.squeeze()

    mask_tensor = torch.clamp(
        mask_tensor,
        0,
        1
    )

    return mask_tensor.numpy()


# ============================================================
# LOCAL ANOMALY HEATMAP
# ============================================================

def generate_local_anomaly_map(
    image,
    feature_extractor,
    normal_mean,
    normal_std
):

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        feature_map = feature_extractor(
            image_tensor
        )

    # --------------------------------------------------------
    # Compare local features with normal features
    # --------------------------------------------------------

    difference = (
        feature_map
        - normal_mean.unsqueeze(0)
    )

    standardized_difference = (
        difference
        / (
            normal_std.unsqueeze(0)
            + 1e-6
        )
    )

    # --------------------------------------------------------
    # Aggregate channels
    # --------------------------------------------------------

    anomaly_map = torch.norm(
        standardized_difference,
        dim=1
    )

    anomaly_map = anomaly_map.squeeze(0)

    # --------------------------------------------------------
    # Resize feature map
    # --------------------------------------------------------

    anomaly_map = (
        anomaly_map
        .unsqueeze(0)
        .unsqueeze(0)
    )

    anomaly_map = F.interpolate(
        anomaly_map,
        size=(256, 256),
        mode="bicubic",
        align_corners=False
    )

    anomaly_map = anomaly_map.squeeze()

    # --------------------------------------------------------
    # Robust normalization
    # --------------------------------------------------------

    low = torch.quantile(
        anomaly_map,
        0.05
    )

    high = torch.quantile(
        anomaly_map,
        0.95
    )

    anomaly_map = (
        anomaly_map - low
    ) / (
        high - low + 1e-8
    )

    anomaly_map = torch.clamp(
        anomaly_map,
        0,
        1
    )

    anomaly_map = anomaly_map.cpu().numpy()

    # --------------------------------------------------------
    # Foreground mask
    # --------------------------------------------------------

    foreground_mask = (
        create_foreground_mask(
            image
        )
    )

    # --------------------------------------------------------
    # Suppress background
    # --------------------------------------------------------

    anomaly_map = (
        anomaly_map
        * foreground_mask
    )

    # --------------------------------------------------------
    # Enhance high anomaly regions
    # --------------------------------------------------------

    anomaly_map = np.power(
        anomaly_map,
        1.5
    )

    return anomaly_map


# ============================================================
# CREATE HEATMAP OVERLAY
# ============================================================

def create_heatmap_overlay(
    image,
    anomaly_map
):

    image_array = np.array(
        image.resize(
            (256, 256)
        )
    )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    # Original image
    ax.imshow(
        image_array
    )

    # Transparent anomaly overlay
    alpha_map = (
        anomaly_map * 0.75
    )

    # Only show stronger anomaly areas
    alpha_map = np.clip(
        alpha_map,
        0,
        0.75
    )

    ax.imshow(
        anomaly_map,
        cmap="jet",
        alpha=alpha_map,
        interpolation="bilinear",
        vmin=0,
        vmax=1
    )

    ax.axis("off")

    ax.set_title(
        "AI Anomaly Localization",
        fontsize=15,
        fontweight="bold"
    )

    fig.tight_layout(
        pad=0
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ System Information"
    )

    st.write(
        "**Model:** ResNet18"
    )

    st.write(
        "**Dataset:** MVTec AD"
    )

    st.write(
        "**Category:** Bottle"
    )

    st.write(
        "**Detection:** Global Feature Distance"
    )

    st.write(
        "**Localization:** Local Feature Analysis"
    )

    st.write(
        "**Input Size:** 256 × 256"
    )

    st.write(
        f"**Device:** {device}"
    )

    st.write(
        f"**Threshold:** "
        f"{CLASSIFICATION_THRESHOLD:.4f}"
    )

    st.divider()

    st.subheader(
        "📊 Validated Performance"
    )

    st.metric(
        "Accuracy",
        "91.57%"
    )

    st.metric(
        "Precision",
        "98.28%"
    )

    st.metric(
        "Recall",
        "90.48%"
    )

    st.metric(
        "F1 Score",
        "94.21%"
    )

    st.divider()

    st.info(
        "The classification model was evaluated "
        "on 83 MVTec AD bottle test images."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.header(
    "📤 Upload Inspection Image"
)

uploaded_file = st.file_uploader(
    "Upload a bottle image for AI quality inspection",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.success(
        "Image uploaded successfully!"
    )

    # --------------------------------------------------------
    # LOAD COMPONENTS
    # --------------------------------------------------------

    model = load_model()

    normal_center = (
        load_normal_center()
    )

    normal_mean, normal_std = (
        load_local_statistics()
    )

    feature_extractor = (
        create_feature_extractor()
    )

    # --------------------------------------------------------
    # RUN INSPECTION
    # --------------------------------------------------------

    with st.spinner(
        "🤖 AI is inspecting the image..."
    ):

        global_score = (
            calculate_global_score(
                image,
                model,
                normal_center
            )
        )

        anomaly_map = (
            generate_local_anomaly_map(
                image,
                feature_extractor,
                normal_mean,
                normal_std
            )
        )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if global_score > CLASSIFICATION_THRESHOLD:

        prediction = (
            "DEFECT DETECTED"
        )

        is_defect = True

    else:

        prediction = (
            "GOOD / NORMAL"
        )

        is_defect = False

    # ========================================================
    # RESULT SECTION
    # ========================================================

    st.subheader(
        "📋 Inspection Result"
    )

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with result_col1:

        if is_defect:

            st.markdown(
                '<div class="status-defect">'
                '🔴 DEFECT DETECTED'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="status-good">'
                '🟢 GOOD / NORMAL'
                '</div>',
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    with result_col2:

        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">'
            f'{global_score:.4f}'
            '</div>'
            '<div class="metric-label">'
            'Anomaly Score'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    with result_col3:

        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">'
            f'{CLASSIFICATION_THRESHOLD:.4f}'
            '</div>'
            '<div class="metric-label">'
            'Validated Threshold'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # ========================================================
    # DECISION ANALYSIS
    # ========================================================

    st.divider()

    st.subheader(
        "📐 Decision Analysis"
    )

    if is_defect:

        st.error(
            f"Anomaly score {global_score:.4f} "
            f"is above the validated threshold "
            f"{CLASSIFICATION_THRESHOLD:.4f}."
        )

    else:

        st.success(
            f"Anomaly score {global_score:.4f} "
            f"is below the validated threshold "
            f"{CLASSIFICATION_THRESHOLD:.4f}."
        )

    # ========================================================
    # ORIGINAL + HEATMAP
    # ========================================================

    st.divider()

    st.subheader(
        "🔥 Defect Localization"
    )

    image_col, heatmap_col = (
        st.columns(2)
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    with image_col:

        st.markdown(
            "### 🖼️ Original Image"
        )

        st.image(
            image.resize(
                (256, 256)
            ),
            use_container_width=True
        )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    with heatmap_col:

        st.markdown(
            "### 🌡️ AI Anomaly Heatmap"
        )

        heatmap_fig = (
            create_heatmap_overlay(
                image,
                anomaly_map
            )
        )

        st.pyplot(
            heatmap_fig,
            use_container_width=True
        )

        plt.close(
            heatmap_fig
        )

    st.caption(
        "Heatmap: blue indicates lower local "
        "feature deviation; yellow/red indicates "
        "stronger anomaly response."
    )

    # ========================================================
    # AI EXPLANATION
    # ========================================================

    st.divider()

    st.subheader(
        "🧠 AI Inspection Explanation"
    )

    if is_defect:

        st.warning(
            """
            The ResNet18 feature representation of
            this image differs from the learned normal
            bottle representation.

            The anomaly heatmap provides visual guidance
            about regions with stronger local feature
            deviations.

            The heatmap is an explanatory localization
            aid and is not a pixel-perfect defect mask.
            """
        )

    else:

        st.success(
            """
            The global feature distance is within the
            learned normal range.

            No significant global anomaly was detected.
            The heatmap shows the spatial distribution
            of local feature differences.
            """
        )

    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    st.divider()

    st.subheader(
        "🔬 Technical Details"
    )

    tech1, tech2, tech3 = (
        st.columns(3)
    )

    with tech1:

        st.markdown(
            """
            <div class="technical-card">

            <div class="technical-card-title">
            🧠 Feature Extraction
            </div>

            <div class="technical-card-text">
            Pretrained ResNet18<br>
            ImageNet feature representation<br>
            512-dimensional global feature vector
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with tech2:

        st.markdown(
            f"""
            <div class="technical-card">

            <div class="technical-card-title">
            📊 Classification
            </div>

            <div class="technical-card-text">
            Euclidean distance from normal feature center<br>
            Validated threshold: {CLASSIFICATION_THRESHOLD:.4f}<br>
            Score &gt; threshold → DEFECT
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with tech3:

        st.markdown(
            """
            <div class="technical-card">

            <div class="technical-card-title">
            🌡️ Localization
            </div>

            <div class="technical-card-text">
            Local feature statistics<br>
            Spatial feature map analysis<br>
            Smooth anomaly heatmap visualization
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

else:

    # ========================================================
    # INITIAL PAGE
    # ========================================================

    st.info(
        "👆 Upload a bottle image above to "
        "begin AI quality inspection."
    )

    st.divider()

    st.header(
        "🏭 About This System"
    )

    st.write(
        """
        This project implements an automated industrial
        surface-defect inspection system using deep
        learning and computer vision.

        A pretrained ResNet18 network is used as a feature
        extractor. Normal bottle images are represented in
        feature space, and a new image is classified by its
        Euclidean distance from the learned normal feature
        center.

        A validated threshold of 4.0173 is used for the
        GOOD / DEFECT decision.

        Local feature analysis is additionally used to
        generate an anomaly heatmap that provides visual
        guidance about potentially abnormal regions.
        """
    )

    st.divider()

    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.subheader(
        "📊 Validated Test Performance"
    )

    p1, p2, p3, p4 = (
        st.columns(4)
    )

    with p1:

        st.metric(
            "Accuracy",
            "91.57%"
        )

    with p2:

        st.metric(
            "Precision",
            "98.28%"
        )

    with p3:

        st.metric(
            "Recall",
            "90.48%"
        )

    with p4:

        st.metric(
            "F1 Score",
            "94.21%"
        )

    st.divider()

    # ========================================================
    # PIPELINE
    # ========================================================

    st.subheader(
        "🔄 System Pipeline"
    )

    pipeline1, pipeline2, pipeline3, pipeline4 = (
        st.columns(4)
    )

    with pipeline1:

        st.markdown(
            """
            <div class="pipeline-card">
            <h3>📷 1</h3>
            <b>Input Image</b>
            <br>
            Bottle image
            </div>
            """,
            unsafe_allow_html=True
        )

    with pipeline2:

        st.markdown(
            """
            <div class="pipeline-card">
            <h3>🧠 2</h3>
            <b>ResNet18</b>
            <br>
            Feature extraction
            </div>
            """,
            unsafe_allow_html=True
        )

    with pipeline3:

        st.markdown(
            """
            <div class="pipeline-card">
            <h3>📊 3</h3>
            <b>Anomaly Detection</b>
            <br>
            Feature distance
            </div>
            """,
            unsafe_allow_html=True
        )

    with pipeline4:

        st.markdown(
            """
            <div class="pipeline-card">
            <h3>🌡️ 4</h3>
            <b>Localization</b>
            <br>
            Anomaly heatmap
            </div>
            """,
            unsafe_allow_html=True
        )