import streamlit as st
import tensorflow as tf
import os
import requests
import traceback
import numpy as np
from PIL import Image
from deep_translator import GoogleTranslator

# Initialize Translator
translator = GoogleTranslator()

# Load model once at import time (faster and avoids re-loading on every prediction)
MODEL_FILENAME = "trained_model.keras"
# Build an absolute path relative to this file so Streamlit's working dir won't break it
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
# Optional model URL (can be set as env var MODEL_URL). Default: raw file from GitHub (public repo).
DEFAULT_MODEL_URL = f"https://raw.githubusercontent.com/chavanarya36/PDD/main/{MODEL_FILENAME}"


def download_model(url, dest_path):
    """Stream-download a file and save to dest_path. Returns True on success."""
    try:
        print(f"Attempting to download model from: {url}")
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print("Model download completed")
        return True
    except Exception as e:
        print(f"Model download failed: {e}")
        return False
model = None
model_load_error = None

def ensure_model_loaded():
    """Try to load the model once and populate model or model_load_error."""
    global model, model_load_error
    if model is not None or model_load_error is not None:
        return
    if not os.path.exists(MODEL_PATH):
        # Try to fetch the model automatically if a URL is provided via env or fallback to raw GitHub URL
        model_url = os.environ.get("MODEL_URL", DEFAULT_MODEL_URL)
        downloaded = False
        try:
            downloaded = download_model(model_url, MODEL_PATH)
        except Exception:
            downloaded = False

        if not downloaded:
            model_load_error = f"Model file not found at: {os.path.abspath(MODEL_PATH)} and automatic download failed (tried {model_url})."
            return
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        model = None
        model_load_error = traceback.format_exc()

# Translation Function
def translate_text(text, target_language):
    return translator.translate(text, source='auto', target=target_language)

# TensorFlow Model Prediction
def model_prediction(test_image):
    ensure_model_loaded()
    if model is None:
        # Return None to the caller and let the UI show the error rather than crashing
        return None
    # Streamlit uploads file-like object; Image.open handles path or file-like
    img = Image.open(test_image).convert('RGB').resize((128, 128))  # Resize the image
    # IMPORTANT: this model expects pixel values in the 0-255 range (see diagnostics).
    input_arr = np.array(img).astype(np.float32)
    input_arr = np.expand_dims(input_arr, axis=0)  # Expand dims to match model input
    prediction = model.predict(input_arr)
    return prediction

# Function to provide treatment suggestions
def suggest_treatment(disease, lang='en'):
    treatments = {
        'Apple___Apple_scab': 'Use fungicide and remove infected leaves.',
        'Apple___Black_rot': 'Prune infected branches and apply appropriate fungicides.',
        'Apple___Cedar_apple_rust': 'Remove nearby juniper trees if possible and apply fungicide.',
        'Apple___healthy': 'No action needed; your plant is healthy!',
        'Blueberry___healthy': 'No action needed; your plant is healthy!',
        'Cherry_(including_sour)___Powdery_mildew': 'Apply fungicides and ensure proper air circulation.',
        'Cherry_(including_sour)___healthy': 'No action needed; your plant is healthy!',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Apply fungicides and practice crop rotation.',
        'Corn_(maize)___Common_rust_': 'Use rust-resistant varieties and apply fungicide if severe.',
        'Corn_(maize)___Northern_Leaf_Blight': 'Use resistant hybrids and apply fungicide if needed.',
        'Corn_(maize)___healthy': 'No action needed; your plant is healthy!',
        'Grape___Black_rot': 'Remove infected leaves and apply fungicides regularly.',
        'Grape___Esca_(Black_Measles)': 'Prune infected vines and avoid water stress.',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Remove infected leaves and apply fungicides as necessary.',
        'Grape___healthy': 'No action needed; your plant is healthy!',
        'Orange___Haunglongbing_(Citrus_greening)': 'Remove infected trees and control psyllid populations.',
        'Peach___Bacterial_spot': 'Apply copper-based bactericides and remove infected fruits.',
        'Peach___healthy': 'No action needed; your plant is healthy!',
        'Pepper,_bell___Bacterial_spot': 'Apply copper-based bactericides and use resistant varieties.',
        'Pepper,_bell___healthy': 'No action needed; your plant is healthy!',
        'Potato___Early_blight': 'Apply fungicides and rotate crops.',
        'Potato___Late_blight': 'Use blight-resistant varieties and apply fungicide immediately.',
        'Potato___healthy': 'No action needed; your plant is healthy!',
        'Raspberry___healthy': 'No action needed; your plant is healthy!',
        'Soybean___healthy': 'No action needed; your plant is healthy!',
        'Squash___Powdery_mildew': 'Use sulfur-based fungicides and increase air circulation.',
        'Strawberry___Leaf_scorch': 'Avoid overhead watering and remove infected leaves.',
        'Strawberry___healthy': 'No action needed; your plant is healthy!',
        'Tomato___Bacterial_spot': 'Use copper-based sprays and avoid overhead irrigation.',
        'Tomato___Early_blight': 'Apply fungicides and practice crop rotation.',
        'Tomato___Late_blight': 'Remove affected plants and apply fungicides immediately.',
        'Tomato___Leaf_Mold': 'Improve air circulation and reduce humidity.',
        'Tomato___Septoria_leaf_spot': 'Remove infected leaves and apply fungicide.',
        'Tomato___Spider_mites Two-spotted_spider_mite': 'Use insecticidal soap or neem oil.',
        'Tomato___Target_Spot': 'Apply fungicides and maintain proper spacing between plants.',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whitefly populations and remove infected plants.',
        'Tomato___Tomato_mosaic_virus': 'Remove infected plants and control aphid populations.',
        'Tomato___healthy': 'No action needed; your plant is healthy!'
    }
    treatment = treatments.get(disease, 'Consult an expert for further advice.')
    if lang != 'en':
        treatment = translate_text(treatment, lang)
    return treatment

# Sidebar
st.sidebar.title("Dashboard")

# Language selection
language = st.sidebar.selectbox("Choose Language", ["English", "Hindi"])
lang_code = 'en' if language == 'English' else 'hi'

# Translation for page titles and content
def t(text):
    return translate_text(text, lang_code) if lang_code != 'en' else text

if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Home"

# Function to handle button clicks
def handle_click(mode):
    st.session_state.app_mode = mode

st.sidebar.button(t("Home"), on_click=handle_click, args=("Home",))
st.sidebar.button(t("About"), on_click=handle_click, args=("About",))
st.sidebar.button(t("Disease Recognition"), on_click=handle_click, args=("Disease Recognition",))

# Main Page
if st.session_state.app_mode == "Home":
    st.header(t("PLANT DISEASE RECOGNITION SYSTEM"))
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test', 'profile.jpg')
    st.image(image_path, use_container_width=True)
    st.markdown(t("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
    """))

# About Project
elif st.session_state.app_mode == "About":
    st.header(t("About Us"))
    st.markdown(t("""
    ### Project Overview
    Our aim is to provide an easy-to-use tool for farmers and gardeners to identify plant diseases quickly and effectively.
    """))

# Prediction Page
elif st.session_state.app_mode == "Disease Recognition":
    st.header(t("Disease Recognition"))
    test_image = st.file_uploader(t("Choose an Image:"), type=["jpg", "jpeg", "png"])

    # If model failed to load, show detailed error and stop
    ensure_model_loaded()
    if model is None:
        st.error(t("Model failed to load."))
        if model_load_error:
            st.code(model_load_error)
        st.stop()

    if test_image is not None:
        img = Image.open(test_image)
        st.image(img, width=400, use_container_width=True)

    # Confidence threshold (UI)
    conf_threshold = st.sidebar.slider('Confidence threshold (%)', 0, 100, 20)

    if st.button(t("Predict")):
        st.snow()
        st.write(t("Analyzing the image..."))
        predictions = model_prediction(test_image)
        if predictions is None:
            st.error(t("Model is not available for prediction. Check model load status above."))
            st.stop()

        preds_arr = np.asarray(predictions)
        # handle shapes like (1, N) or (N,)
        if preds_arr.ndim == 2 and preds_arr.shape[0] == 1:
            probs = preds_arr[0]
        else:
            probs = preds_arr.flatten()

        result_index = int(np.argmax(probs))
        confidence_levels = probs
        # Global class list (must match the training order)
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
            'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]

        # Top-3 predictions
        top3_idx = np.argsort(confidence_levels)[-3:][::-1]
        st.success(t(f"Top predictions:"))
        for idx in top3_idx:
            cls = class_name[idx]
            conf = confidence_levels[idx] * 100
            st.write(f"- {cls}: {conf:.2f}%")

        # Primary predicted class and treatment
        predicted_class = class_name[result_index]
        confidence = confidence_levels[result_index] * 100
        if confidence < conf_threshold:
            st.warning(t(f"Low confidence ({confidence:.2f}%). The model is unsure — consider uploading a clearer image or using multiple images."))

        st.write(t(f"Primary prediction: {predicted_class} ({confidence:.2f}%)"))
        treatment = suggest_treatment(predicted_class, lang_code)
        st.write(f"**{t('Suggested Treatment:')}** {treatment}")
