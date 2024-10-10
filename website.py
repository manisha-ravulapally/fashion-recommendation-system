import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from annoy import AnnoyIndex  # Annoy library
from numpy.linalg import norm
import time
# Load pre-trained data
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Define dimensions based on the feature list
dimensions = feature_list.shape[1]

# Build and load Annoy index
index = AnnoyIndex(dimensions, metric='euclidean')
for i in range(len(feature_list)):
    index.add_item(i, feature_list[i])
index.build(10)  # Build the index with 10 trees

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# Define pages
def landing_page():
    page_bg="""
    <style>
    [data-testid ="stAppViewContainer"]{
    background-image: url("https://media.istockphoto.com/id/1193802723/photo/beautiful-asian-woman-carrying-colorful-bags-shopping-online-with-mobile-phone.jpg?s=612x612&w=0&k=20&c=_Uinu-GgmyoS-ZOJ9nHRr6yQOdvZm_fohZ2DqluRx8I=");
    background-size: cover;
    }
    </style>
    """
    st.markdown(page_bg,unsafe_allow_html=True)
    st.title('Welcome to the Fashion Recommender System')
    st.write("Upload your fashion image, and we will recommend similar items for you.")
    if st.button('Get Started'):
        st.session_state.page = 'upload'  # Navigate to the upload page

def upload_page():
    page_bg = """
        <style>
        [data-testid ="stAppViewContainer"]{
        background-image: url("https://htmlcolorcodes.com/assets/images/colors/light-purple-color-solid-background-1920x1080.png");
        background-size: cover;
        }
        </style>
        """
    st.markdown(page_bg, unsafe_allow_html=True)
    st.title('Upload Your Fashion Image')
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            # Display uploaded image
            display_image = Image.open(uploaded_file)
            resized_image = display_image.resize((300, 300))
            st.image(resized_image, caption='Uploaded Image')

            # Add a button to show recommendations
            if st.button("Show Recommendations"):
                # Feature extraction
                features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

                # Recommendation
                indices = recommend(features, index)

                # Navigate to recommendations page
                st.session_state.page = 'recommendation'
                st.session_state.uploaded_file = uploaded_file
                st.session_state.indices = indices
            if st.button("Go Back"):
                st.session_state.page = 'landing'

def recommendation_page():
    page_bg = """
            <style>
            [data-testid ="stAppViewContainer"]{
            background-image: url("https://htmlcolorcodes.com/assets/images/colors/light-purple-color-solid-background-1920x1080.png");
            background-size: cover;
            }
            </style>
            """
    st.markdown(page_bg, unsafe_allow_html=True)

    uploaded_file = st.session_state.uploaded_file
    indices = st.session_state.indices



    # Display recommended images
    st.subheader("Recommended Items:")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.image(resize_image(filenames[indices[i]]))
    if st.button("Go Back"):
        st.session_state.page = 'upload'

# Helper functions
def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('uploads', exist_ok=True)
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, index, n=5):
    start_time = time.time()
    indices = index.get_nns_by_vector(features, n)
    end_time = time.time()
    print(f"Time taken by Annoy: {end_time - start_time} seconds")
    return indices

def resize_image(img_path, size=(500, 500)):
    img = Image.open(img_path)
    img = img.resize(size, Image.LANCZOS)
    return img

# Main execution
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

if st.session_state.page == 'landing':
    landing_page()
elif st.session_state.page == 'upload':
    upload_page()
elif st.session_state.page == 'recommendation':
    recommendation_page()