import streamlit as st
import tensorflow as tf
import os
import traceback
import numpy as np
from PIL import Image
from deep_translator import GoogleTranslator
import base64

# Translation helpers: cache translator instance and translated strings
@st.cache_resource(show_spinner=False)
def get_translator(target_language: str):
    """Cache and return a GoogleTranslator instance per target language."""
    return GoogleTranslator(source='en', target=target_language)

@st.cache_data(show_spinner=False)
def translate_text(text, target_language):
    """Translate text to target language using a cached translator (fast)."""
    if target_language == 'en' or not text:
        return text
    try:
        translator = get_translator(target_language)
        return translator.translate(text)
    except Exception as e:
        # If translation fails, return original text
        print(f"Translation error: {e}")
        return text

# Custom CSS for world-class UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for theming */
    :root {
        --primary-color: #10b981;
        --primary-dark: #059669;
        --primary-light: #34d399;
        --secondary-color: #6366f1;
        --accent-color: #f59e0b;
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-card: rgba(30, 41, 59, 0.8);
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --border-radius: 16px;
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.1);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.2);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(100, 102, 241, 0.2);
    }
    
    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 12px 24px;
        font-weight: 600;
        font-size: 15px;
        margin: 8px 0;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--secondary-color) 100%);
    }
    
    [data-testid="stSidebar"] .stButton button:active {
        transform: translateY(0);
    }
    
    /* Headings */
    h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-light) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        margin-bottom: 1rem;
        text-align: center;
        animation: fadeInDown 0.8s ease;
    }
    
    h2 {
        font-family: 'Poppins', sans-serif;
        color: var(--text-primary);
        font-weight: 600;
        font-size: 2rem;
        margin: 2rem 0 1rem 0;
    }
    
    h3 {
        font-family: 'Poppins', sans-serif;
        color: var(--primary-light);
        font-weight: 500;
        font-size: 1.5rem;
    }
    
    /* Card styling */
    .card {
        background: var(--bg-card);
        backdrop-filter: blur(10px);
        border-radius: var(--border-radius);
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-md);
        border: 1px solid rgba(100, 102, 241, 0.1);
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-lg);
        border-color: rgba(100, 102, 241, 0.3);
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
        border-radius: var(--border-radius);
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(16, 185, 129, 0.2);
        height: 100%;
    }
    
    .feature-card:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%);
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Upload section */
    [data-testid="stFileUploader"] {
        background: var(--bg-card);
        border: 2px dashed var(--primary-color);
        border-radius: var(--border-radius);
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary-light);
        background: rgba(16, 185, 129, 0.05);
        transform: scale(1.02);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 14px 32px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-md);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4);
        background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary-color) 100%);
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo, .stWarning {
        background: var(--bg-card);
        border-radius: var(--border-radius);
        border-left: 4px solid var(--primary-color);
        padding: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .stWarning {
        border-left-color: var(--warning);
    }
    
    /* Progress and metrics */
    .stMetric {
        background: var(--bg-card);
        padding: 1.5rem;
        border-radius: var(--border-radius);
        border: 1px solid rgba(16, 185, 129, 0.2);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        border-color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    /* Images */
    img {
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-lg);
        transition: all 0.3s ease;
    }
    
    img:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 40px rgba(16, 185, 129, 0.3);
    }
    
    /* Selectbox and slider */
    .stSelectbox, .stSlider {
        background: var(--bg-card);
        border-radius: var(--border-radius);
        padding: 0.5rem;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    /* Confidence bar */
    .confidence-bar {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
        height: 24px;
        margin: 8px 0;
        overflow: hidden;
        position: relative;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--primary-light) 100%);
        border-radius: 10px;
        transition: width 1s ease;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding: 0 12px;
        color: white;
        font-weight: 600;
        font-size: 12px;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(16, 185, 129, 0.3);
        animation: fadeInUp 0.5s ease;
    }
    
    /* Text styling */
    p {
        color: var(--text-secondary);
        line-height: 1.8;
        font-size: 1.05rem;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        margin: 4px;
        box-shadow: var(--shadow-sm);
    }
    
    /* Hero section */
    .hero {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        animation: fadeInDown 0.8s ease;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, var(--primary-color) 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(100, 102, 241, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# Load model once at import time (faster and avoids re-loading on every prediction)
MODEL_FILENAME = "trained_model.keras"
# Build an absolute path relative to this file so Streamlit's working dir won't break it
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
model = None
model_load_error = None

@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str):
    """Load and cache the Keras model once across reruns and sessions."""
    # compile=False speeds up loading and avoids optimizer deserialization warnings
    return tf.keras.models.load_model(model_path, compile=False)

def ensure_model_loaded():
    """Try to load the model once and populate model or model_load_error."""
    global model, model_load_error
    if model is not None or model_load_error is not None:
        return
    if not os.path.exists(MODEL_PATH):
        model_load_error = f"Model file not found at: {os.path.abspath(MODEL_PATH)}"
        return
    try:
        # Use cached loader to avoid reloading on every rerun
        model = load_model_cached(MODEL_PATH)
    except Exception:
        model = None
        model_load_error = traceback.format_exc()

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
st.sidebar.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>🌿 Navigation</h2>", unsafe_allow_html=True)

# Language selection with flags
language_options = {
    "English": "🇬🇧",
    "Hindi": "🇮🇳"
}
selected_language = st.sidebar.selectbox(
    "🌐 Choose Language", 
    list(language_options.keys()),
    format_func=lambda x: f"{language_options[x]} {x}"
)
lang_code = 'en' if selected_language == 'English' else 'hi'

# Translation helper function - translates all text
def t(text):
    """Translate text to selected language"""
    return translate_text(text, lang_code) if lang_code != 'en' else text

if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Home"
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False

# Function to handle button clicks
def handle_click(mode):
    st.session_state.app_mode = mode

# Navigation buttons with icons
nav_buttons = {
    "Home": "🏠",
    "About": "ℹ️",
    "Disease Recognition": "🔬"
}

st.sidebar.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
# Model preload/status controls
st.sidebar.markdown(f"<h3 style='text-align: center;'>🧠 {t('AI Model')}</h3>", unsafe_allow_html=True)
preload_col1, preload_col2 = st.sidebar.columns([1, 1])
with preload_col1:
    if st.button(f"⚡ {t('Preload Model')}"):
        with st.spinner(t('Loading AI model... First load may take up to 30–60 seconds')):
            ensure_model_loaded()
            st.session_state.model_ready = model is not None
            if model is not None:
                st.sidebar.success(t('Model loaded and ready'))
            else:
                st.sidebar.error(t('Model failed to load'))
with preload_col2:
    status = t('Ready') if st.session_state.model_ready and model is not None else t('Not Loaded')
    badge_color = '#10b981' if status == t('Ready') else '#f59e0b'
    st.sidebar.markdown(
        f"<div style='text-align:center; padding:8px; border-radius:12px; border:1px solid rgba(255,255,255,0.1);'><strong>"+
        t('Status')+f": </strong><span style='color:{badge_color}'>{status}</span></div>",
        unsafe_allow_html=True,
    )
for page, icon in nav_buttons.items():
    if st.sidebar.button(f"{icon} {t(page)}", key=f"nav_{page}"):
        handle_click(page)

# Main Page
if st.session_state.app_mode == "Home":
    # Hero section
    st.markdown(f"""
    <div class='hero'>
        <h1>🌿 {t('PLANT DISEASE RECOGNITION SYSTEM')}</h1>
        <p style='font-size: 1.3rem; color: var(--text-secondary); margin-top: 1rem;'>
            {t('Powered by Advanced AI • Fast • Accurate • Easy to Use')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main image
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test', 'profile.jpg')
    if os.path.exists(image_path):
        # Replace deprecated use_container_width with width='stretch'
        st.image(image_path, width='stretch')
    
    # Welcome message
    welcome_text = t('Together, let\'s protect our crops and ensure a healthier, greener harvest!')
    st.markdown(f"""
    <div class='card'>
        <h2 style='text-align: center; margin-top: 0;'>{t('Welcome to the Future of Plant Health')} 🚀</h2>
        <p style='text-align: center; font-size: 1.1rem;'>
            {t('Our mission is to help farmers, gardeners, and plant enthusiasts identify plant diseases efficiently and accurately. Upload an image of a plant, and our advanced AI system will analyze it to detect any signs of diseases in seconds.')}
        </p>
        <p style='text-align: center; font-size: 1.1rem; color: var(--primary-light);'>
            {welcome_text} 🌱
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    st.markdown(f"<h2 style='text-align: center; margin-top: 3rem;'>✨ {t('Key Features')}</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='feature-card'>
            <div class='feature-icon'>⚡</div>
            <h3 style='color: var(--text-primary);'>{t('Instant Analysis')}</h3>
            <p>{t('Get results in seconds with our optimized AI model trained on thousands of plant images.')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='feature-card'>
            <div class='feature-icon'>🎯</div>
            <h3 style='color: var(--text-primary);'>{t('High Accuracy')}</h3>
            <p>{t('96%+ validation accuracy across 38 different plant disease categories.')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='feature-card'>
            <div class='feature-icon'>💊</div>
            <h3 style='color: var(--text-primary);'>{t('Treatment Advice')}</h3>
            <p>{t('Receive actionable treatment recommendations for detected diseases.')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Statistics
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center;'>📊 {t('Our Impact')}</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='card' style='text-align: center;'>
            <h2 style='color: var(--primary-light); margin: 0; font-size: 2.5rem;'>38</h2>
            <p style='margin: 0.5rem 0 0 0;'>{t('Disease Types')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='card' style='text-align: center;'>
            <h2 style='color: var(--primary-light); margin: 0; font-size: 2.5rem;'>96%</h2>
            <p style='margin: 0.5rem 0 0 0;'>{t('Accuracy')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='card' style='text-align: center;'>
            <h2 style='color: var(--primary-light); margin: 0; font-size: 2.5rem;'>&lt;2s</h2>
            <p style='margin: 0.5rem 0 0 0;'>{t('Analysis Time')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='card' style='text-align: center;'>
            <h2 style='color: var(--primary-light); margin: 0; font-size: 2.5rem;'>24/7</h2>
            <p style='margin: 0.5rem 0 0 0;'>{t('Available')}</p>
        </div>
        """, unsafe_allow_html=True)

# About Project
elif st.session_state.app_mode == "About":
    st.markdown(f"""
    <div class='hero'>
        <h1>ℹ️ {t('About Our Project')}</h1>
        <p style='font-size: 1.2rem; color: var(--text-secondary); margin-top: 1rem;'>
            {t('Empowering Agriculture with Artificial Intelligence')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project overview
    vision_text = t('Our aim is to provide an easy-to-use, accessible tool for farmers and gardeners worldwide to identify plant diseases quickly and effectively. By leveraging cutting-edge deep learning technology, we\'re making professional-grade plant disease diagnosis available to everyone.')
    st.markdown(f"""
    <div class='card'>
        <h2>🎯 {t('Project Vision')}</h2>
        <p>
            {vision_text}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technology stack
    st.markdown(f"""
    <div class='card'>
        <h2>🔧 {t('Technology Stack')}</h2>
        <div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 1rem;'>
            <span class='badge'>🧠 TensorFlow 2.17</span>
            <span class='badge'>🎨 Streamlit</span>
            <span class='badge'>🖼️ Keras 3.5</span>
            <span class='badge'>📷 Computer Vision</span>
            <span class='badge'>🌐 Deep Learning</span>
            <span class='badge'>🔬 PIL/Pillow</span>
            <span class='badge'>🌍 Google Translator</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model details
    st.markdown(f"""
    <div class='card'>
        <h2>🤖 {t('Model Architecture')}</h2>
        <p><strong>{t('Input')}:</strong> {t('128×128×3 RGB images (raw 0-255 pixel values)')}</p>
        <p><strong>{t('Output')}:</strong> {t('38-class softmax predictions with confidence scores')}</p>
        <p><strong>{t('Training')}:</strong> {t('Optimized with data augmentation and dropout regularization')}</p>
        <p><strong>{t('Validation Accuracy')}:</strong> <span style='color: var(--primary-light); font-weight: 600;'>96.4%</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Supported plants
    st.markdown(f"""
    <div class='card'>
        <h2>🌿 {t('Supported Plants')}</h2>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-top: 1rem;'>
            <div style='text-align: center; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 10px;'>
                <div style='font-size: 2rem;'>🍎</div>
                <p style='margin: 0.5rem 0 0 0; font-weight: 600;'>{t('Apple')}</p>
            </div>
            <div style='text-align: center; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 10px;'>
                <div style='font-size: 2rem;'>🌽</div>
                <p style='margin: 0.5rem 0 0 0; font-weight: 600;'>{t('Corn')}</p>
            </div>
            <div style='text-align: center; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 10px;'>
                <div style='font-size: 2rem;'>🍇</div>
                <p style='margin: 0.5rem 0 0 0; font-weight: 600;'>{t('Grape')}</p>
            </div>
            <div style='text-align: center; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 10px;'>
                <div style='font-size: 2rem;'>🥔</div>
                <p style='margin: 0.5rem 0 0 0; font-weight: 600;'>{t('Potato')}</p>
            </div>
            <div style='text-align: center; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 10px;'>
                <div style='font-size: 2rem;'>🍅</div>
                <p style='margin: 0.5rem 0 0 0; font-weight: 600;'>{t('Tomato')}</p>
            </div>
            <div style='text-align: center; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 10px;'>
                <div style='font-size: 2rem;'>🍑</div>
                <p style='margin: 0.5rem 0 0 0; font-weight: 600;'>{t('Peach')}</p>
            </div>
            <div style='text-align: center; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 10px;'>
                <div style='font-size: 2rem;'>🌶️</div>
                <p style='margin: 0.5rem 0 0 0; font-weight: 600;'>{t('Pepper')}</p>
            </div>
            <div style='text-align: center; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 10px;'>
                <div style='font-size: 2rem;'>🍓</div>
                <p style='margin: 0.5rem 0 0 0; font-weight: 600;'>{t('Strawberry')}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # How it works
    st.markdown(f"""
    <div class='card'>
        <h2>⚙️ {t('How It Works')}</h2>
        <ol style='font-size: 1.05rem; line-height: 2;'>
            <li><strong>{t('Upload Image')}:</strong> {t('Take or upload a clear photo of the plant leaf')}</li>
            <li><strong>{t('Preprocessing')}:</strong> {t('Image is resized and normalized for optimal analysis')}</li>
            <li><strong>{t('AI Analysis')}:</strong> {t('Deep learning model processes the image through neural networks')}</li>
            <li><strong>{t('Prediction')}:</strong> {t('System identifies disease with confidence scores')}</li>
            <li><strong>{t('Treatment')}:</strong> {t('Receive actionable recommendations to treat the disease')}</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Prediction Page
elif st.session_state.app_mode == "Disease Recognition":
    st.markdown(f"""
    <div class='hero'>
        <h1>🔬 {t('Disease Recognition')}</h1>
        <p style='font-size: 1.2rem; color: var(--text-secondary); margin-top: 1rem;'>
            {t('Upload a plant image for instant AI-powered disease detection')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload section
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>📤 {t('Upload Plant Image')}</h3>", unsafe_allow_html=True)
    test_image = st.file_uploader(
        t("Choose a clear image of the plant leaf (JPG, JPEG, PNG)"), 
        type=["jpg", "jpeg", "png"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Hint about first-run latency (defer heavy model load until analysis)
    st.markdown(
        f"<p style='text-align:center; color: var(--text-secondary);'>⏱️ {t('Note: The AI model loads on first analysis and may take up to 30–60 seconds the first time. Subsequent analyses are instant.')}</p>",
        unsafe_allow_html=True,
    )

    # Image preview
    if test_image is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>🖼️ {t('Image Preview')}</h3>", unsafe_allow_html=True)
        img = Image.open(test_image)
        
        # Display image with custom styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Replace deprecated use_container_width with width='stretch'
            st.image(img, width='stretch')
        
        # Image info
        st.markdown(f"""
        <p style='text-align: center; color: var(--text-secondary); margin-top: 1rem;'>
            📐 {t('Size')}: {img.size[0]} × {img.size[1]} pixels | 
            🎨 {t('Mode')}: {img.mode} | 
            📦 {t('Format')}: {img.format}
        </p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Confidence threshold in sidebar
    st.sidebar.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<h3 style='text-align: center;'>⚙️ {t('Settings')}</h3>", unsafe_allow_html=True)
    conf_threshold = st.sidebar.slider(
        f'🎯 {t("Confidence Threshold")} (%)', 
        min_value=0, 
        max_value=100, 
        value=20,
        help=t("Minimum confidence level for predictions")
    )

    # Predict button
    if test_image is not None:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button(f"🔍 {t('ANALYZE NOW')}", use_container_width=True)
        
        if predict_button:
            # Loading animation and lazy model load
            with st.spinner(t('� Loading AI model and analyzing... (first run may take up to 30–60 seconds)')):
                st.snow()
                # Ensure model is loaded only when actually needed
                ensure_model_loaded()
                if model is None:
                    st.markdown(f"""
                    <div class='card' style='border-left: 4px solid var(--error);'>
                        <h3 style='color: var(--error);'>⚠️ {t('Model Loading Error')}</h3>
                        <p>{t('The AI model failed to load. Please check the console for details.')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if model_load_error:
                        st.code(model_load_error)
                    st.stop()

                predictions = model_prediction(test_image)
                
                if predictions is None:
                    st.error(t("Model is not available for prediction. Check model load status above."))
                    st.stop()

                preds_arr = np.asarray(predictions)
                if preds_arr.ndim == 2 and preds_arr.shape[0] == 1:
                    probs = preds_arr[0]
                else:
                    probs = preds_arr.flatten()

                result_index = int(np.argmax(probs))
                confidence_levels = probs
                
                # Global class list
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

                # Results section
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("<h2 style='text-align: center;'>📊 Analysis Results</h2>", unsafe_allow_html=True)
                
                # Top prediction
                predicted_class = class_name[result_index]
                confidence = confidence_levels[result_index] * 100
                
                # Primary result card
                if confidence >= conf_threshold:
                    st.markdown(f"""
                    <div class='result-card' style='border: 2px solid var(--primary-color);'>
                        <h2 style='text-align: center; color: var(--primary-light); margin-top: 0;'>
                            🎯 Primary Detection
                        </h2>
                        <h3 style='text-align: center; color: var(--text-primary); font-size: 1.8rem;'>
                            {predicted_class.replace('___', ' - ').replace('_', ' ')}
                        </h3>
                        <div class='confidence-bar'>
                            <div class='confidence-fill' style='width: {confidence}%;'>
                                {confidence:.2f}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='card' style='border-left: 4px solid var(--warning);'>
                        <h3 style='color: var(--warning);'>⚠️ {t('Low Confidence Detection')}</h3>
                        <p>{t('The model detected')} <strong>{predicted_class.replace('___', ' - ').replace('_', ' ')}</strong> 
                        {t('with only')} <strong>{confidence:.2f}%</strong> {t('confidence')}.</p>
                        <p>💡 <strong>{t('Suggestions')}:</strong></p>
                        <ul>
                            <li>{t('Upload a clearer, well-lit image')}</li>
                            <li>{t('Ensure the leaf is in focus')}</li>
                            <li>{t('Try multiple images of the same plant')}</li>
                            <li>{t('Minimize background clutter')}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Top 3 predictions
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center;'>🏆 {t('Top 3 Predictions')}</h3>", unsafe_allow_html=True)
                
                top3_idx = np.argsort(confidence_levels)[-3:][::-1]
                
                for rank, idx in enumerate(top3_idx, 1):
                    cls = class_name[idx]
                    conf = confidence_levels[idx] * 100
                    medal = ["🥇", "🥈", "🥉"][rank-1]
                    
                    st.markdown(f"""
                    <div style='background: rgba(16, 185, 129, 0.05); border-radius: 10px; padding: 1rem; margin: 0.5rem 0;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-size: 1.5rem;'>{medal}</span>
                            <span style='flex: 1; margin: 0 1rem; font-weight: 600; color: var(--text-primary);'>
                                {cls.replace('___', ' - ').replace('_', ' ')}
                            </span>
                            <span style='color: var(--primary-light); font-weight: 700; font-size: 1.1rem;'>
                                {conf:.2f}%
                            </span>
                        </div>
                        <div class='confidence-bar' style='height: 8px; margin-top: 0.5rem;'>
                            <div class='confidence-fill' style='width: {conf}%; height: 8px;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Treatment recommendation
                treatment = suggest_treatment(predicted_class, lang_code)
                treatment_header = translate_text("Suggested Treatment", lang_code) if lang_code != 'en' else "Suggested Treatment"
                st.markdown(f"""
                <div class='card' style='border-left: 4px solid var(--primary-color);'>
                    <h3 style='color: var(--primary-light);'>💊 {treatment_header}</h3>
                    <p style='font-size: 1.1rem; line-height: 1.8;'>{treatment}</p>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # Instructions when no image
        tip_text = t('Tip: Multiple angles of the same plant give better insights!')
        st.markdown(f"""
        <div class='card'>
            <h3 style='text-align: center; color: var(--primary-light);'>📋 {t('Instructions')}</h3>
            <ol style='font-size: 1.05rem; line-height: 2;'>
                <li>{t('Take a clear photo of the plant leaf showing symptoms')}</li>
                <li>{t('Ensure good lighting and focus')}</li>
                <li>{t('Avoid excessive background elements')}</li>
                <li>{t('Upload the image using the file picker above')}</li>
                <li>{t('Click "Analyze Now" to get instant results')}</li>
            </ol>
            <p style='text-align: center; margin-top: 2rem; color: var(--text-secondary);'>
                💡 <strong>{tip_text}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
footer_made_with = t('Made with ❤️ for farmers and plant enthusiasts worldwide')
st.markdown(f"""
<div class='footer'>
    <p>🌿 <strong>{t('Plant Disease Recognition System')}</strong> | {t('Powered by AI & Deep Learning')}</p>
    <p style='margin-top: 0.5rem;'>{footer_made_with}</p>
    <p style='margin-top: 0.5rem; font-size: 0.85rem;'>
        TensorFlow • Keras • Streamlit • Computer Vision
    </p>
</div>
""", unsafe_allow_html=True)

