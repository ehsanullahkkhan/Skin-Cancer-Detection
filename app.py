import json
import os
import requests
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🩹"
)

MODEL_URL = "https://huggingface.co/Ehsikhan/Skin-Cancer-Detection/resolve/main/best_skin_cancer_model.keras"
MODEL_PATH = "/tmp/best_skin_cancer_model.keras"


@st.cache_resource
def load_artifacts():

    # Step 1: Download model
    if not os.path.exists(MODEL_PATH):

        st.info("⏳ Downloading model from Hugging Face...")

        response = requests.get(
            MODEL_URL,
            stream=True,
            timeout=300
        )

        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    # Step 2: Load model
    st.info("⏳ Loading AI model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    # Step 3: Load class names
    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    return model, class_names


model, class_names = load_artifacts()

IMG_SIZE = (224, 224)

st.title("Skin Cancer Detection (Benign vs Malignant)")

st.caption(
    "Educational demo only — NOT a medical diagnostic tool. "
    "Always consult a dermatologist."
)

uploaded = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    img = image.resize(IMG_SIZE)

    arr = np.array(img).astype("float32") / 255.0

    arr = np.expand_dims(arr, axis=0)

    prob = float(
        model.predict(arr, verbose=0).r
