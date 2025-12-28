import streamlit as st
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="COVID-19 X-ray Detection",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("COVID-19 Detection from Chest X-ray")
st.write("Upload a chest X-ray image to test the image pipeline")

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PROCESS IMAGE ----------------
if uploaded_file is not None:
    # Load image using PIL
    image = Image.open(uploaded_file).convert("RGB")

    # Show original image
    st.subheader("Uploaded X-ray")
    st.image(image, width='stretch')

    # Preprocess (same as CNN input shape)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    st.subheader("Preprocessing Status")
    st.success("✅ Image loaded and preprocessed successfully")
    st.info("CNN model is NOT loaded yet. Training will be done in Google Colab.")

else:
    st.warning("Please upload a chest X-ray image to continue.")
