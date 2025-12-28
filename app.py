import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="COVID-19 X-ray Detection",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("COVID-19 Detection from Chest X-ray")
st.write("Upload a chest X-ray image for COVID-19 prediction")

# ---------------- MODEL DOWNLOAD + LOAD ----------------
MODEL_URL = "https://drive.google.com/uc?id=1tkv4C77aBONnRAv_u74kVz8twp-Aj_-X"
MODEL_PATH = "covid_xray_cnn_final.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... Please wait (first run only)."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PROCESS IMAGE ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded X-ray")
    st.image(image, width="stretch")

    # Preprocessing (MUST match training)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # ---------------- PREDICTION ----------------
    prediction = model.predict(image_array)[0][0]

    st.subheader("Prediction Result")

    if prediction < 0.5:
        st.error(
            f"🦠 COVID-19 Detected\n\nConfidence: {(1 - prediction) * 100:.2f}%"
        )
    else:
        st.success(
            f"✅ Normal Chest X-ray\n\nConfidence: {prediction * 100:.2f}%"
        )

else:
    st.info("Please upload a chest X-ray image to continue.")
