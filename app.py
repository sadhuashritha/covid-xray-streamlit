import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
import os

st.set_page_config(page_title="COVID-19 X-Ray Detection", layout="centered")

st.title("🫁 COVID-19 Chest X-Ray Detection")

MODEL_PATH = "covid_xray_cnn_final.keras"
IMG_SIZE = 224

# ---------- DEBUG CHECK ----------
st.write("Model file exists:", os.path.exists(MODEL_PATH))

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH, compile=False)

try:
    with st.spinner("Loading model..."):
        model = load_model()
    st.success("Model loaded successfully")
except Exception as e:
    st.error("Model failed to load")
    st.exception(e)
    st.stop()

# ---------- IMAGE UPLOAD ----------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess_image(image)

    with st.spinner("Predicting..."):
        prediction = model.predict(img)

    prob = float(prediction[0][0])

    if prob >= 0.5:
        st.error(f"🦠 COVID-19 Detected ({prob:.2%})")
    else:
        st.success(f"✅ Normal ({(1-prob):.2%})")

st.markdown("---")
st.caption("For educational purposes only.")
