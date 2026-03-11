import streamlit as st
import numpy as np
from PIL import Image
from face_detection import extract_face
from utils.predict import predict

st.title("🧠 Deepfake Image Detector")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:

    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)

    st.image(image)

    face = extract_face(image_np)

    if face is not None:

        label, confidence = predict(face)

        if label == "Fake":
            st.error(f"Fake Image ({confidence:.2f})")
        else:
            st.success(f"Real Image ({confidence:.2f})")

    else:
        st.warning("No face detected")