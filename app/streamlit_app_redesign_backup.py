
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

PROJECT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT / "models"

DATASET_DIR = PROJECT / "data" / "raw" / "EuroSAT_RGB"

SAMPLE_DIR = PROJECT / "app" / "sample_images"

CNN_PATH = MODEL_DIR / "best_cnn_v2_model.keras"

RESNET_PATH = MODEL_DIR / "best_resnet18_model.keras"


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


MODEL_ACCURACY = {
    "CNN V2": 86.81,
    "ResNet18": 76.15
}


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EuroSAT | Earth Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f7f9fc;
    color: #10243b;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1350px;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e6edf4;
}

section[data-testid="stSidebar"] h1 {
    font-family: 'Manrope', sans-serif;
    color: #102b46;
    font-size: 25px;
    font-weight: 800;
}

section[data-testid="stSidebar"] p {
    color: #66788a;
}

/* HEADINGS */

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: #102b46 !important;
    font-weight: 800 !important;
}

h1 {
    letter-spacing: -1.5px;
}

p {
    color: #607286;
}

/* HERO */

.hero {
    background: linear-gradient(
        115deg,
        #ffffff 0%,
        #f0f6fc 100%
    );
    border: 1px solid #e2ebf4;
    border-radius: 25px;
    padding: 55px 45px;
    display: grid;
    grid-template-columns: 1.2fr 0.8fr;
    align-items: center;
    min-height: 400px;
    overflow: hidden;
    position: relative;
}

.eyebrow {
    color: #2186ae;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2.5px;
    margin-bottom: 23px;
}

.hero h1 {
    color: #102b46 !important;
    font-size: 49px;
    line-height: 1.13;
    letter-spacing: -2px;
    margin-bottom: 20px;
}

.hero p {
    font-size: 15px;
    line-height: 1.9;
    max-width: 520px;
    color: #62758a;
    margin-bottom: 25px;
}

.pill {
    display: inline-block;
    border: 1px solid #d6e6f2;
    background: #ffffff;
    color: #286d91;
    border-radius: 30px;
    padding: 9px 14px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.6px;
    margin: 4px 5px 4px 0;
}

/* EARTH ILLUSTRATION */

.earth-area {
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    height: 310px;
}

.globe {
    width: 230px;
    height: 230px;
    border-radius: 50%;
    position: relative;
    background:
        radial-gradient(
            ellipse at 35% 30%,
            #8bd4dd 0%,
            #2786a8 28%,
            #125078 62%,
            #082a4a 100%
        );
    box-shadow:
        inset -25px -15px 35px rgba(0,0,0,0.35),
        inset 12px 8px 25px rgba(255,255,255,0.3),
        0 20px 55px rgba(29,105,145,0.25);
    overflow: hidden;
}

.globe:before {
    content: "";
    position: absolute;
    width: 95px;
    height: 80px;
    top: 37px;
    left: 38px;
    background: #74c59c;
    border-radius: 48% 28% 55% 35%;
    transform: rotate(-22deg);
    box-shadow:
        45px 65px 0 -8px #78cba0,
        -10px 100px 0 -28px #75bd93;
}

.globe:after {
    content: "";
    position: absolute;
    width: 45px;
    height: 90px;
    top: 90px;
    left: 140px;
    background: #68b88e;
    border-radius: 40% 60% 45% 55%;
    transform: rotate(25deg);
}

.orbit {
    width: 330px;
    height: 115px;
    border: 1.5px solid rgba(32,133,177,0.5);
    border-radius: 50%;
    position: absolute;
    transform: rotate(-25deg);
}

.orbit:after {
    content: "";
    position: absolute;
    width: 12px;
    height: 12px;
    background: #24a6cf;
    border-radius: 50%;
    top: 4px;
    right: 47px;
    box-shadow: 0 0 18px #24a6cf;
}

/* CARDS */

.info-card {
    background: #ffffff;
    border: 1px solid #e5edf4;
    border-radius: 18px;
    padding: 23px;
    height: 100%;
}

.info-card h3 {
    font-size: 17px;
    margin-bottom: 8px;
}

.info-card p {
    font-size: 13px;
    line-height: 1.7;
}

.section-label {
    color: #2585a8;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 9px;
}

.metric-card {
    background: white;
    border: 1px solid #e5edf4;
    border-radius: 16px;
    padding: 20px;
}

.metric-label {
    color: #75869a;
    font-size: 12px;
    font-weight: 600;
}

.metric-number {
    color: #123c5c;
    font-family: 'Manrope', sans-serif;
    font-size: 29px;
    font-weight: 800;
    margin-top: 7px;
}

/* BUTTONS */

.stButton > button {
    border-radius: 10px;
    border: 1px solid #187ba7;
    background: #187ba7;
    color: white;
    font-weight: 700;
    padding: 10px 22px;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #105b81;
    border-color: #105b81;
    color: white;
}

/* UPLOAD */

[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #a9c7db;
    border-radius: 15px;
    padding: 15px;
}

/* DIVIDER */

hr {
    border-color: #e5edf4;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #8494a5;
    font-size: 12px;
    padding: 30px 0 10px 0;
}

@media(max-width: 800px) {
    .hero {
        grid-template-columns: 1fr;
        padding: 30px 22px;
    }

    .hero h1 {
        font-size: 36px;
    }

    .earth-area {
        height: 260px;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model(model_path):

    return tf.keras.models.load_model(
        str(model_path),
        compile=False
    )


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((64, 64))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image, model_path):

    model = load_model(model_path)

    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    predictions = np.asarray(predictions).flatten()

    if len(predictions) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected 10 class scores, but received "
            f"{len(predictions)}."
        )

    # Convert outputs to probabilities if necessary.
    if (
        np.any(predictions < 0)
        or not np.isclose(np.sum(predictions), 1.0, atol=0.02)
    ):
        predictions = tf.nn.softmax(predictions).numpy()

    predicted_index = int(np.argmax(predictions))

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(predictions[predicted_index]) * 100

    top_indices = np.argsort(predictions)[::-1][:3]

    top_results = [
        (
            CLASS_NAMES[index],
            float(predictions[index]) * 100
        )
        for index in top_indices
    ]

    return predicted_class, confidence, top_results, predictions


# =========================================================
# DATASET SAMPLE IMAGE
# =========================================================

def get_sample_image(category):

    extensions = ["*.jpg", "*.jpeg", "*.png"]

    search_paths = [
        SAMPLE_DIR / category,
        DATASET_DIR / category
    ]

    for folder in search_paths:

        if folder.exists():

            for extension in extensions:

                images = list(folder.glob(extension))

                if images:
                    return images[0]

    return None


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("# 🌍 EuroSAT")

    st.caption("EARTH INTELLIGENCE PLATFORM")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "Prediction",
            "Models",
            "Dataset",
            "About"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### PROJECT DETAILS")

    st.caption("Dataset: EuroSAT RGB")

    st.caption("Input: 64 × 64 RGB")

    st.caption("Classes: 10")

    st.caption("Models: CNN + ResNet18")

    st.divider()

    st.caption("Remote Sensing Image Classification")


# =========================================================
# HOME PAGE
# =========================================================

if page == "Home":

    st.markdown("""
    <div class="hero">

        <div>

            <div class="eyebrow">
                EARTH OBSERVATION / AI INTELLIGENCE
            </div>

            <h1>
                See the land.<br>
                Understand the planet.
            </h1>

            <p>
                Explore satellite imagery with deep learning.
                Classify land-cover patterns, inspect model
                confidence and compare neural networks trained
                on the EuroSAT RGB dataset.
            </p>

            <span class="pill">10 LAND-COVER CLASSES</span>
            <span class="pill">RGB IMAGERY</span>
            <span class="pill">CNN + RESNET18</span>

        </div>

        <div class="earth-area">
            <div class="orbit"></div>
            <div class="globe"></div>
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown(
        '<div class="section-label">PLATFORM OVERVIEW</div>',
        unsafe_allow_html=True
    )

    st.header("Explore Earth through AI")

    st.write(
        "A deep-learning platform designed to understand "
        "different land-cover patterns from satellite images."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🛰️ Image Prediction</h3>
            <p>
                Upload satellite imagery and identify
                its land-cover category using trained
                deep-learning models.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🧠 Model Intelligence</h3>
            <p>
                Explore and compare CNN and ResNet18
                models using their test accuracy
                and prediction results.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>🌎 Dataset Explorer</h3>
            <p>
                Discover the ten EuroSAT land-cover
                categories through dataset statistics
                and sample images.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("TOTAL IMAGES", "27,000"),
        ("LAND-COVER CLASSES", "10"),
        ("IMAGE SIZE", "64 × 64"),
        ("INPUT FORMAT", "RGB")
    ]

    for column, (label, value) in zip(
        [c1, c2, c3, c4],
        metrics
    ):

        with column:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-number">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "Prediction":

    st.markdown(
        '<div class="section-label">AI IMAGE ANALYSIS</div>',
        unsafe_allow_html=True
    )

    st.title("Satellite Image Prediction")

    st.write(
        "Upload a satellite image and let the trained model "
        "identify its land-cover category."
    )

    st.info(
        "For best compatibility, use satellite images similar "
        "to the EuroSAT RGB dataset. Images are resized to 64 × 64."
    )

    col1, col2 = st.columns([1, 1])

    with col1:

        model_choice = st.selectbox(
            "Choose Deep Learning Model",
            ["CNN V2", "ResNet18"]
        )

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"]
        )

    with col2:

        st.markdown("#### SUPPORTED CLASSES")

        st.write(
            "Annual Crop · Forest · Herbaceous Vegetation · "
            "Highway · Industrial · Pasture · Permanent Crop · "
            "Residential · River · Sea/Lake"
        )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("### Image Preview")

        st.image(
            image,
            caption="Uploaded Satellite Image",
            width=350
        )

        if st.button(
            "🚀 Analyze Image",
            use_container_width=True
        ):

            model_path = (
                CNN_PATH
                if model_choice == "CNN V2"
                else RESNET_PATH
            )

            if not model_path.exists():

                st.error(
                    f"Model file not found: {model_path}"
                )

            else:

                try:

                    with st.spinner(
                        "Analyzing satellite image..."
                    ):

                        result = predict_image(
                            image,
                            model_path
                        )

                    predicted_class, confidence, top_results, scores = result

                    st.session_state["last_prediction"] = {
                        "class": predicted_class,
                        "confidence": confidence,
                        "top_results": top_results,
                        "model": model_choice
                    }

                    st.success("Image analysis completed!")

                    st.markdown("## Prediction Result")

                    r1, r2 = st.columns(2)

                    with r1:

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">
                                    PREDICTED LAND-COVER CLASS
                                </div>
                                <div class="metric-number">
                                    {predicted_class}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with r2:

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">
                                    MODEL CONFIDENCE
                                </div>
                                <div class="metric-number">
                                    {confidence:.2f}%
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.write("")

                    st.markdown("### Confidence Level")

                    st.progress(
                        min(max(confidence / 100, 0.0), 1.0)
                    )

                    if confidence < 60:

                        st.warning(
                            "Low prediction confidence. "
                            "The model is uncertain about this image. "
                            "It may be unfamiliar or outside the "
                            "EuroSAT dataset."
                        )

                    st.markdown("### Top 3 Predictions")

                    for name, score in top_results:

                        st.write(
                            f"**{name}** — {score:.2f}%"
                        )

                        st.progress(
                            min(max(score / 100, 0.0), 1.0)
                        )

                    st.markdown("### All Class Probabilities")

                    chart_data = pd.DataFrame({
                        "Land Cover": CLASS_NAMES,
                        "Confidence": scores * 100
                    })

                    st.bar_chart(
                        chart_data.set_index("Land Cover")
                    )

                except Exception as error:

                    st.error(
                        f"Prediction failed: {error}"
                    )

    else:

        st.markdown("""
        <div class="info-card">
            <h3>📡 Ready for image analysis</h3>
            <p>
                Upload an RGB image to begin land-cover
                classification.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Note: A confidence score below 60% is a warning, "
        "not a guaranteed detection of non-satellite images."
    )


# =========================================================
# MODELS PAGE
# =========================================================

elif page == "Models":

    st.markdown(
        '<div class="section-label">MODEL PERFORMANCE</div>',
        unsafe_allow_html=True
    )

    st.title("Deep Learning Models")

    st.write(
        "Compare the performance of the two trained "
        "models used in this project."
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
        <div class="info-card">
            <h3>🧠 Custom CNN V2</h3>
            <p>
                A custom Convolutional Neural Network
                trained to recognize satellite image
                land-cover patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.metric(
            "Test Accuracy",
            "86.81%"
        )

        if CNN_PATH.exists():
            st.success("CNN model file found")
        else:
            st.error("CNN model file missing")

    with c2:

        st.markdown("""
        <div class="info-card">
            <h3>🔬 ResNet18</h3>
            <p>
                A residual neural network architecture
                used for image classification and
                model performance comparison.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.metric(
            "Test Accuracy",
            "76.15%"
        )

        if RESNET_PATH.exists():
            st.success("ResNet18 model file found")
        else:
            st.error("ResNet18 model file missing")

    st.divider()

    st.subheader("Accuracy Comparison")

    comparison = pd.DataFrame({
        "Model": ["CNN V2", "ResNet18"],
        "Accuracy": [86.81, 76.15]
    })

    st.bar_chart(
        comparison.set_index("Model")
    )

    st.write(
        "**Observation:** CNN V2 achieved higher test accuracy "
        "than ResNet18 in the recorded project results."
    )

    st.caption(
        "These are the previously recorded test accuracies, "
        "not newly calculated measurements."
    )


# =========================================================
# DATASET PAGE
# =========================================================

elif page == "Dataset":

    st.markdown(
        '<div class="section-label">DATA EXPLORATION</div>',
        unsafe_allow_html=True
    )

    st.title("EuroSAT Dataset Explorer")

    st.write(
        "Explore the satellite image dataset and its "
        "ten land-cover categories."
    )

    total_images = sum(CLASS_COUNTS.values())

    a, b, c = st.columns(3)

    a.metric("Total Images", f"{total_images:,}")

    b.metric("Number of Classes", "10")

    c.metric("Image Resolution", "64 × 64 RGB")

    st.divider()

    st.subheader("Land-Cover Distribution")

    count_df = pd.DataFrame({
        "Class": list(CLASS_COUNTS.keys()),
        "Images": list(CLASS_COUNTS.values())
    })

    st.bar_chart(
        count_df.set_index("Class")
    )

    st.subheader("Explore Individual Categories")

    selected_class = st.selectbox(
        "Select a land-cover category",
        CLASS_NAMES
    )

    sample_path = get_sample_image(selected_class)

    if sample_path:

        sample_image = Image.open(sample_path)

        left, right = st.columns([1, 1])

        with left:

            st.image(
                sample_image,
                caption=selected_class,
                width=350
            )

        with right:

            st.markdown(
                f"### {selected_class}"
            )

            st.write(
                f"Images in dataset: "
                f"{CLASS_COUNTS[selected_class]:,}"
            )

            st.write(
                "This category is part of the "
                "EuroSAT RGB land-cover classification dataset."
            )

    else:

        st.warning(
            "No sample image found for this category. "
            "Check the dataset and sample_images folders."
        )

    st.divider()

    st.subheader("Dataset Split")

    split_data = pd.DataFrame({
        "Dataset": ["Training", "Validation", "Testing"],
        "Images": [18900, 5400, 2700]
    })

    st.bar_chart(
        split_data.set_index("Dataset")
    )

    st.caption(
        "Recorded dataset split: 70% training, "
        "20% validation and 10% testing."
    )


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "About":

    st.markdown(
        '<div class="section-label">PROJECT INFORMATION</div>',
        unsafe_allow_html=True
    )

    st.title("About This Project")

    st.write(
        "Remote Sensing Image Classification Using EuroSAT"
    )

    st.markdown("""
    <div class="info-card">

        <h3>🌍 Project Overview</h3>

        <p>
            This project uses deep-learning techniques
            to classify satellite images into different
            land-cover categories using the EuroSAT RGB dataset.
        </p>

        <p>
            The system compares a custom Convolutional Neural
            Network (CNN V2) and ResNet18 to understand their
            performance in satellite image classification.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.subheader("Technologies Used")

    technologies = [
        "Python",
        "TensorFlow",
        "Keras",
        "Streamlit",
        "NumPy",
        "Pandas",
        "Pillow",
        "EuroSAT RGB Dataset"
    ]

    cols = st.columns(4)

    for index, technology in enumerate(technologies):

        with cols[index % 4]:

            st.markdown(
                f'<span class="pill">{technology}</span>',
                unsafe_allow_html=True
            )

    st.divider()

    st.subheader("Project Objectives")

    st.markdown("""
    - Classify satellite images into ten land-cover classes.
    - Implement CNN and ResNet18 architectures.
    - Compare trained model performance.
    - Display prediction confidence and class probabilities.
    - Build an interactive Streamlit application.
    """)

    st.subheader("Future Enhancements")

    st.markdown("""
    - Improve detection of non-satellite images.
    - Add more advanced model architectures.
    - Explore explainable AI.
    - Expand image analysis capabilities.
    """)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    EARTH OBSERVATION / AI INTELLIGENCE
    <br>
    Remote Sensing Image Classification Using EuroSAT
    <br>
    Built with Python, TensorFlow and Streamlit
</div>
""", unsafe_allow_html=True)
