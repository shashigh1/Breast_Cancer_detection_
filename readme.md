# 🧠 Breast Cancer Detection using VGG16

A deep learning web application to classify breast histopathology images as **Cancer** or **Not Cancer** using a pre-trained VGG16 model.

## 🚀 Features

- Upload any JPG/PNG histopathology image.
- Model predicts whether it's cancerous or not.
- Built using TensorFlow, Keras, and Streamlit.
- Lightweight deployment: Model downloaded dynamically using `gdown`.

## 🧩 Tech Stack

- Python
- TensorFlow + Keras
- Streamlit
- Google Drive + gdown (for downloading model)

## 📦 Setup

```bash
git clone https://github.com/yourusername/Breast_Cancer_detection.git
cd Breast_Cancer_detection
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
