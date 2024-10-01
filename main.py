import streamlit as st
import numpy as np


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .st-emotion-cache-mnu3yk {visibility:hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.sidebar.title("Dashboard")

if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Home"

# Function to handle button clicks


def handle_click(mode):
    st.session_state.app_mode = mode


st.sidebar.button("Home", on_click=handle_click, args=("Home",))
st.sidebar.button("About", on_click=handle_click, args=("About",))
st.sidebar.button("Disease Recognition", on_click=handle_click,
                  args=("Disease Recognition",))


# app_mode = st.sidebar.selectbox(
#     "Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if (st.session_state.app_mode == "Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    # image_path = "home_page.jpeg"
    # st.image(image_path, use_column_width=True)
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
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

# Prediction Page
elif (st.session_state.app_mode == "Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if (st.button("Show Image")):
        st.image(test_image, width=4, use_column_width=True)
    # Predict button
    if (st.button("Predict")):
        # st.snow()
        st.write("Our Prediction")
