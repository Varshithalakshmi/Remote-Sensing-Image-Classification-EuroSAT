
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

# ==========================================
# EURO SAT CLASSES
# ==========================================

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

# ==========================================
# MODEL DETAILS
# ==========================================

MODEL_FILES = {
    "Custom CNN V1": "best_cnn_v2_model.keras",
    "ResNet18": "best_resnet18_model.keras"
}

MODEL_ACCURACY = {
    "Custom CNN V1": 86.81,
    "ResNet18": 76.15
}

CONFIDENCE_THRESHOLD = 60.0

# ==========================================
# DATASET DETAILS
# ==========================================

CLASS_COUNTS = {
    "AnnualCrop": 3000,
    "Forest": 3000,
    "HerbaceousVegetation": 3000,
    "Highway": 2500,
    "Industrial": 2500,
    "Pasture": 2000,
    "PermanentCrop": 2500,
    "Residential": 3000,
    "River": 2500,
    "SeaLake": 3000
}

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="EuroSAT | Satellite AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #f3f7fa;
}

[data-testid="stSidebar"] {
    background-color: #102a43;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #102a43;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 16px;
    color: #627d98;
}

.hero {
    background: linear-gradient(120deg, #102a43, #176b87);
    padding: 38px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 36px;
}

.hero p {
    color: #d9edf5;
    font-size: 17px;
}

.stat-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e1eaf0;
    box-shadow: 0 3px 12px rgba(16,42,67,0.05);
}

.stat-label {
    color: #627d98;
    font-size: 14px;
}

.stat-value {
    color: #102a43;
    font-size: 27px;
    font-weight: 800;
}

.section-title {
    color: #102a43;
    font-size: 24px;
    font-weight: 700;
}

div.stButton > button {
    background-color: #087e8b;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 22px;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #05636e;
    color: white;
}

.footer {
    text-align: center;
    color: #627d98;
    padding: 25px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("# 🛰️ EuroSAT")
    st.caption("Satellite Image Intelligence")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Home", "Prediction", "Models", "Dataset"]
    )

    st.divider()

    st.markdown("### Project Information")
    st.write("**Dataset:** EuroSAT RGB")
    st.write("**Classes:** 10")
    st.write("**Input:** 64 × 64 RGB")

    st.divider()

    st.caption("Remote Sensing Image Classification")

# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>🛰️ Satellite Image Classification</h1>
        <p>
        Explore land-cover classification using deep learning
        and the EuroSAT RGB satellite image dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<p class="section-title">Project Overview</p>',
        unsafe_allow_html=True
    )

    st.write("""
    This project uses deep learning models to identify different
    land-cover categories from satellite images.

    Upload an image on the Prediction page to classify it using
    the trained models.
    """)

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("Total Images", "27,000"),
        ("Land-cover Classes", "10"),
        ("Best Test Accuracy", "86.81%"),
        ("Image Input Size", "64 × 64")
    ]

    for col, (label, value) in zip(
        [c1, c2, c3, c4], metrics
    ):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    st.markdown(
        '<p class="section-title">Land-cover Categories</p>',
        unsafe_allow_html=True
    )

    cols = st.columns(5)

    for i, name in enumerate(CLASS_NAMES):
        with cols[i % 5]:
            st.info(name)

    st.markdown(
        '<p class="section-title">Available Deep Learning Models</p>',
        unsafe_allow_html=True
    )

    m1, m2 = st.columns(2)

    with m1:
        st.markdown("### 🧠 Custom CNN V1")
        st.write("Improved custom Convolutional Neural Network.")
        st.metric("Test Accuracy", "86.81%")

    with m2:
        st.markdown("### 🔷 ResNet18")
        st.write("Residual Neural Network for image classification.")
        st.metric("Test Accuracy", "76.15%")

# ==========================================
# PREDICTION PAGE
# ==========================================

elif page == "Prediction":

    st.markdown(
        '<p class="main-title">Image Prediction</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Upload a satellite image and identify its land-cover category.</p>',
        unsafe_allow_html=True
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        model_choice = st.selectbox(
            "Select Deep Learning Model",
            list(MODEL_FILES.keys())
        )

        uploaded_file = st.file_uploader(
            "Upload Satellite Image",
            type=["jpg", "jpeg", "png"]
        )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            with right:
                st.markdown("### Image Preview")
                st.image(image, use_container_width=True)
                st.caption(
                    f"Image dimensions: {image.width} × {image.height}"
                )

            st.divider()

            if st.button("🔍 Analyze Image", type="primary"):

                model_path = MODEL_DIR / MODEL_FILES[model_choice]

                if not model_path.exists():
                    st.error(f"Model file not found: {model_path}")

                else:
                    with st.spinner("Analyzing satellite image..."):

                        model = load_model(str(model_path))

                        resized_image = image.resize(
                            (64, 64),
                            Image.Resampling.BILINEAR
                        )

                        image_array = np.asarray(
                            resized_image,
                            dtype=np.float32
                        ) / 255.0

                        image_array = np.expand_dims(
                            image_array,
                            axis=0
                        )

                        predictions = model.predict(
                            image_array,
                            verbose=0
                        )[0]

                        if len(predictions) != len(CLASS_NAMES):
                            st.error(
                                "Model output does not match the 10 EuroSAT classes."
                            )
                            st.stop()

                        predicted_index = int(np.argmax(predictions))

                        predicted_class = CLASS_NAMES[predicted_index]

                        confidence = float(
                            predictions[predicted_index]
                        ) * 100

                        top_indices = np.argsort(
                            predictions
                        )[::-1][:3]

                    st.divider()

                    if confidence < CONFIDENCE_THRESHOLD:

                        st.error("❌ Image Not Recognized")

                        st.warning(
                            "The model has low confidence in this prediction. "
                            "The image may not belong to a supported EuroSAT class."
                        )

                        st.info(
                            "Supported classes: " + ", ".join(CLASS_NAMES)
                        )

                        st.metric(
                            "Highest Model Confidence",
                            f"{confidence:.2f}%"
                        )

                    else:

                        st.success("Image classification completed!")

                        st.markdown("## Prediction Result")

                        r1, r2 = st.columns(2)

                        with r1:
                            st.metric(
                                "Predicted Category",
                                predicted_class
                            )

                        with r2:
                            st.metric(
                                "Confidence Score",
                                f"{confidence:.2f}%"
                            )

                        st.divider()

                        st.markdown("### Top 3 Predictions")

                        for index in top_indices:

                            name = CLASS_NAMES[index]
                            score = float(predictions[index])

                            st.write(
                                f"**{name} — {score * 100:.2f}%**"
                            )

                            st.progress(
                                float(np.clip(score, 0, 1))
                            )

                        with st.expander(
                            "View All Class Prediction Scores"
                        ):

                            for name, score in zip(
                                CLASS_NAMES, predictions
                            ):
                                st.write(
                                    f"{name}: {float(score) * 100:.2f}%"
                                )

                                st.progress(
                                    float(np.clip(score, 0, 1))
                                )

                        st.caption(
                            "Confidence is the model's estimated probability, "
                            "not a guarantee of correctness."
                        )

        except Exception as error:
            st.error(f"Image processing error: {error}")

    else:
        with right:
            st.info(
                "Upload an image to see its preview and start prediction."
            )

    st.warning(
        "These models were trained on EuroSAT satellite RGB images. "
        "Predictions on ordinary photographs may be unreliable."
    )

# ==========================================
# MODELS PAGE
# ==========================================

elif page == "Models":

    st.markdown(
        '<p class="main-title">Deep Learning Models</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Comparison of trained models using the same EuroSAT test set.</p>',
        unsafe_allow_html=True
    )

    st.divider()

    a, b = st.columns(2)

    with a:
        st.markdown("### 🧠 Custom CNN V1")
        st.metric("Test Accuracy", "86.81%")

        st.write("**Architecture details**")
        st.write("- Convolutional layers")
        st.write("- Batch Normalization")
        st.write("- Data Augmentation")
        st.write("- Global Average Pooling")
        st.write("- Dropout and Dense layers")

    with b:
        st.markdown("### 🔷 ResNet18")
        st.metric("Test Accuracy", "76.15%")

        st.write("**Architecture details**")
        st.write("- Residual learning architecture")
        st.write("- Convolutional layers")
        st.write("- Residual connections")
        st.write("- Global Average Pooling")
        st.write("- Fully connected output layer")

    st.divider()

    st.markdown("### Model Accuracy Comparison")

    for name, accuracy in MODEL_ACCURACY.items():
        st.write(f"**{name} — {accuracy:.2f}%**")
        st.progress(accuracy / 100)

    improvement = (
        MODEL_ACCURACY["Custom CNN V1"]
        - MODEL_ACCURACY["ResNet18"]
    )

    st.success(
        f"Custom CNN V1 achieved {improvement:.2f} percentage "
        "points higher test accuracy than ResNet18."
    )

    st.caption(
        "Both models were evaluated on the same 2,700-image EuroSAT test set."
    )

# ==========================================
# DATASET PAGE
# ==========================================

elif page == "Dataset":

    st.markdown(
        '<p class="main-title">EuroSAT Dataset</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Explore the dataset used to train and evaluate the classification models.</p>',
        unsafe_allow_html=True
    )

    st.divider()

    d1, d2, d3 = st.columns(3)

    d1.metric("Total Images", "27,000")
    d2.metric("Number of Classes", "10")
    d3.metric("Image Format", "RGB")

    st.divider()

    st.markdown("### Class Distribution")

    for name, count in CLASS_COUNTS.items():

        col1, col2 = st.columns([2, 4])

        with col1:
            st.write(f"**{name}**")

        with col2:
            st.progress(count / 3000)
            st.caption(f"{count:,} images")

    st.divider()

    st.markdown("### Dataset Split")

    split1, split2, split3 = st.columns(3)

    split1.metric("Training Images", "18,900")
    split1.caption("70% of dataset")

    split2.metric("Validation Images", "5,400")
    split2.caption("20% of dataset")

    split3.metric("Testing Images", "2,700")
    split3.caption("10% of dataset")

    st.markdown("### Dataset Split Distribution")

    st.write("Training — 70%")
    st.progress(0.70)

    st.write("Validation — 20%")
    st.progress(0.20)

    st.write("Testing — 10%")
    st.progress(0.10)

    st.divider()

    st.markdown("### Dataset Information")

    st.write("""
    - Dataset: EuroSAT RGB
    - Total images: 27,000
    - Number of land-cover categories: 10
    - Input image dimensions: 64 × 64
    - Image channels: 3 (RGB)
    - Training, validation and testing split: 70%, 20%, 10%
    """)

    st.info(
        "The dataset contains satellite RGB images representing "
        "ten different land-cover categories."
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown("""
<div class="footer">
    🛰️ Remote Sensing Image Classification Using EuroSAT
    <br><br>
    Deep Learning | Custom CNN V1 | ResNet18
</div>
""", unsafe_allow_html=True)
