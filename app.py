import os
import pandas as pd
import numpy as np
import requests
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from PIL import Image
import streamlit as st

vgg16_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=vgg16_model.input, outputs=vgg16_model.output)

def prepare_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)

def extract_features(image_path):
    preprocessed_image = prepare_image(image_path)
    features = model.predict(preprocessed_image)
    return features.flatten()

def extract_features_from_folder(folder_path):
    feature_list = []
    image_names = []
    
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
            features = extract_features(image_path)
            feature_list.append(features)
            image_names.append(image_name)
    
    return feature_list, image_names
    
def find_top_similar_images(query_image, feature_list, image_names, folder_path, top_n=5):
    query_features = extract_features(query_image)
    similarities = []
    similar_images = []
    
    for i, features in enumerate(feature_list):
        similarity = cosine_similarity([query_features], [features])[0][0]
        similarities.append((image_names[i], similarity))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    for image_name, _ in similarities[:top_n]:
        similar_images.append(os.path.join(folder_path, image_name))
    return similar_images

def main():
    st.set_page_config(page_title="Image Match Finder", layout="wide")
    st.markdown("""
        <style>
            .main { background-color: #f0f0f5; }
            .header { text-align: center; margin-bottom: 20px; }
            .logo { width: 80px; height: 80px; margin-right: 20px; }
            .footer { position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); font-size: 14px; color: #FF69B4; font-weight: bold; background-color: #3498db; padding: 10px; width: 100%; text-align: center; }
            .uploaded-image { width: 200px; height: auto; margin-bottom: 10px; }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("Logo.webp", width=80) 
    with col2:
        st.title("Image Match Finder")
    
    st.subheader("Step 1: Reading Pre-uploaded CSV File")
    csv_path = "DataSheet.csv"
    if not os.path.exists(csv_path):
        st.error("CSV file not found!")
        return
    
    df = pd.read_csv(csv_path)
    st.write("CSV file loaded successfully!")
    
    st.subheader("Step 2: Downloading Images from Links")
    output_dir = "downloaded_images"
    os.makedirs(output_dir, exist_ok=True)
    
    if len(os.listdir(output_dir)) == 0: 
        st.write("Downloading images...")
        for _, row in df.iterrows():
            image_url = row['Image Link']
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                output_path = os.path.join(output_dir, f"{row['Product ID']}.jpg")
                with open(output_path, "wb") as file:
                    file.write(response.content)
        st.success(f"Images downloaded to the folder: {output_dir}")
    else:
        st.info("Images already downloaded.")

    st.subheader("Step 3: Extracting Features")
    if 'feature_list' not in st.session_state:
        st.write("Extracting features from images...")
        feature_list, image_names = extract_features_from_folder(output_dir)
        st.session_state.feature_list = feature_list
        st.session_state.image_names = image_names
        st.success("Features extracted successfully!")
    else:
        st.write("Features are already extracted.")

    st.subheader("Step 4: Upload Query Image")
    uploaded_query_image = st.file_uploader("Upload an image to find similar images", type=["png", "jpg", "jpeg"])

    if uploaded_query_image is not None:
        query_image_path = os.path.join("temp_query_image.jpg")
        with open(query_image_path, "wb") as f:
            f.write(uploaded_query_image.getbuffer())
        
        st.image(Image.open(query_image_path), caption="Uploaded Image", width=200)

        st.write("Finding similar images...")
        top_similar_images = find_top_similar_images(query_image_path, st.session_state.feature_list, st.session_state.image_names, output_dir)

        st.write("Top 5 Similar Images:")
        col1, col2 = st.columns(2)

        for i, image_path in enumerate(top_similar_images):
            col = col1 if i % 2 == 0 else col2
            col.image(image_path, use_column_width=True, width=150)

    st.markdown('<div class="footer">Developed by Mashal Fatima</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
