import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gdown
import os

# Define model path
model_path = "g16_model.keras"

# Download model from Google Drive if not already present
if not os.path.exists(model_path):
    gdown.download(id="14l0VGiU3a_YShnT8AUTvepZMPSKMXuB_", output=model_path, quiet=False)

# Load the model
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = {
    0: "Not Cancer",
    1: "Cancer"
}

# Streamlit app
st.title("🧬 Breast Cancer Image Classification")
st.write("Upload a breast histopathology image to detect cancer.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    probability = prediction[0][0]  # assuming shape is (1, 1)

    class_index = 1 if probability >= 0.5 else 0
    confidence = probability * 100 if class_index == 1 else (1 - probability) * 100

    st.write(f"### 🧾 Prediction: **{class_labels[class_index]}**")
    st.write(f"### 📊 Confidence: **{confidence:.2f}%**")
    st.write(f"🧪 Raw model output: {prediction[0][0]:.4f}")
