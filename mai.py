import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from deep_translator import GoogleTranslator

# Initialize Translator
translator = GoogleTranslator()

# Translation Function
def translate_text(text, target_language):
    return translator.translate(text, source='auto', target=target_language)

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(r"C:\Users\chava\OneDrive\Desktop\PDD\Plant-Disease-Detection\trained_model.keras")
    img = Image.open(test_image).resize((128, 128))  # Resize the image
    input_arr = np.array(img) / 255.0  # Normalize the image
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
    image_path = "C:/Users/chava/Plant-Disease-Detection/test/profile.jpg"
    st.image(image_path, use_column_width=True)
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

    if test_image is not None:
        img = Image.open(test_image)
        st.image(img, width=400, use_column_width=True)

    if st.button(t("Predict")):
        st.snow()
        st.write(t("Analyzing the image..."))
        predictions = model_prediction(test_image)
        result_index = np.argmax(predictions)
        confidence_levels = predictions[0]

        # Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                      ]
        
        predicted_class = class_name[result_index]
        confidence = confidence_levels[result_index] * 100

        st.success(t(f"Model is predicting: {predicted_class} with {confidence:.2f}% confidence"))

        # Suggest treatment
        treatment = suggest_treatment(predicted_class, lang_code)
        st.write(f"**{t('Suggested Treatment:')}** {treatment}")
