import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

@st.cache_resource
def load_my_model():
    return load_model("brain_tumor_model.h5")

model = load_my_model()

# === Load Model ===
model = load_model("brain_tumor_model.h5")

CATEGORIES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
IMG_SIZE = 100

# === Prediction Function ===
def predict_image(image):
    img = np.array(image.convert("L"))  # Convert to grayscale
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

    prediction = model.predict(img)

    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index] * 100

    return CATEGORIES[class_index], confidence


# === Streamlit UI ===
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI scan image to detect brain tumor type.")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded MRI Scan",
        use_container_width=True
    )

    st.markdown("---")

    if st.button("🔍 Predict"):

        result, confidence = predict_image(image)

        if result == "No Tumor":
            st.success(
                f"✅ Prediction: {result} ({confidence:.2f}%)"
            )
        else:
            st.error(
                f"⚠️ Prediction: {result} ({confidence:.2f}%)"
            )