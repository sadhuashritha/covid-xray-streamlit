import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="COVID-19 X-ray Detection",
    layout="centered"
)

st.title("🩻 COVID-19 Detection from Chest X-ray")
st.write("Upload a chest X-ray image to predict COVID-19")

MODEL_PATH = "covid_xray_cnn_final.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width=300)

    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]

    if prediction < 0.5:
        st.error(f"🦠 COVID-19 Detected\nConfidence: {(1 - prediction) * 100:.2f}%")
    else:
        st.success(f"✅ Normal X-ray\nConfidence: {prediction * 100:.2f}%")
