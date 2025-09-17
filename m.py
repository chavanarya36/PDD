import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(r"C:\Users\chava\OneDrive\Desktop\PDD\Plant-Disease-Detection\trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    return prediction

# Function to provide treatment suggestions
def suggest_treatment(disease):
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
        'Potato___Early_blight': 'Apply fungicides and rotate crops to prevent infection.',
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
        'Tomato___Leaf_Mold': 'Ensure good air circulation and apply fungicide if needed.',
        'Tomato___Septoria_leaf_spot': 'Remove infected leaves and apply fungicide regularly.',
        'Tomato___Spider_mites Two-spotted_spider_mite': 'Use miticides and encourage beneficial predators.',
        'Tomato___Target_Spot': 'Apply fungicides and remove infected leaves.',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Use resistant varieties and control whitefly populations.',
        'Tomato___Tomato_mosaic_virus': 'Remove infected plants and sanitize tools.',
        'Tomato___healthy': 'No action needed; your plant is healthy!'
    }
    return treatments.get(disease, 'Consult an expert for further advice.')


# Sidebar
st.sidebar.title("Dashboard")

if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Home"

# Function to handle button clicks
def handle_click(mode):
    st.session_state.app_mode = mode

st.sidebar.button("Home", on_click=handle_click, args=("Home",))
st.sidebar.button("About", on_click=handle_click, args=("About",))
st.sidebar.button("Disease Recognition", on_click=handle_click, args=("Disease Recognition",))

# Main Page
if (st.session_state.app_mode == "Home"):

    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path ="C:/Users/chava/Plant-Disease-Detection/test/profile.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif (st.session_state.app_mode == "About"):

    st.header("About Us")
    st.markdown("""
                ### Project Overview
                Our aim is to provide an easy-to-use tool for farmers and gardeners to identify
                 plant diseases quickly and effectively.

                ### Meet the Team
                - *Vinay Pawar*: Model Builder
                  - Developed the machine learning model for plant disease detection.

                - *Himanshu Pathak*: Model Trainer
                  - Trained the model to ensure high accuracy and reliability in predictions.

                - *Arya Chavan*: UI Developer
                  - Developed the user interface to provide a seamless experience for users.

               

                ### Why Choose Us?
                Our system utilizes state-of-the-art machine learning techniques for accurate disease detection and offers a user-friendly experience.
                """)

# Prediction Page
elif st.session_state.app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, width=400, use_column_width=True)

    if st.button("Predict"):
        st.snow()
        st.write("Analyzing the image...")
        predictions = model_prediction(test_image)
        result_index = np.argmax(predictions)
        confidence_levels = predictions[0]
        
        # Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        
        predicted_class = class_name[result_index]
        confidence = confidence_levels[result_index] * 100
        
        st.success(f"Model is predicting: {predicted_class} with {confidence:.2f}% confidence")

        # Suggest treatment
        treatment = suggest_treatment(predicted_class)
        st.write(f"**Suggested Treatment:** {treatment}")

       
        
