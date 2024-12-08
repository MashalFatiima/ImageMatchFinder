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

# Load pre-trained VGG16 model (without the top layer)
vgg16_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=vgg16_model.input, outputs=vgg16_model.output)

# Function to prepare an image
def prepare_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)

# Function to extract features using VGG16
def extract_features(image_path):
    preprocessed_image = prepare_image(image_path)
    features = model.predict(preprocessed_image)
    return features.flatten()

# Function to extract features from all images in a folder
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

# Function to find the top 5 similar images
def find_top_similar_images(query_image, feature_list, image_names, folder_path, top_n=5):
    query_features = extract_features(query_image)
    similarities = []
    
    for i, features in enumerate(feature_list):
        similarity = cosine_similarity([query_features], [features])[0][0]
        similarities.append((image_names[i], similarity))
    
    # Sort by similarity score in descending order
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Get the paths of the top similar images
    similar_images = []
    for image_name, _ in similarities[:top_n]:
        similar_images.append(os.path.join(folder_path, image_name))
    
    return similar_images

# Streamlit app
def main():
    # Set a nice background color for the app
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #f4f6f9;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    # Display Logo
    st.image("Logo.webp", width=200, caption="Logo", use_column_width=False)  # Adjust the width of your logo
    
    st.title("Image Similarity Finder")
    st.markdown('<p style="color: #3498db;">Developed by FATIMA NOOR</p>', unsafe_allow_html=True)
    
    # Step 1: Read the pre-uploaded CSV file
    st.subheader("Step 1: Reading Pre-uploaded CSV File")
    csv_path = "DataSheet.csv"  # Make sure your CSV is named DataSheet.csv
    if not os.path.exists(csv_path):
        st.error("CSV file not found!")
        return
    
    df = pd.read_csv(csv_path)
    st.write("CSV file loaded successfully!")
    
    # Step 2: Download images from the links and save in a folder
    st.subheader("Step 2: Downloading Images from Links")
    output_dir = "downloaded_images"
    os.makedirs(output_dir, exist_ok=True)
    
    if len(os.listdir(output_dir)) == 0:  # Only download if the folder is empty
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

    # Step 3: Extract features for downloaded images (only once)
    st.subheader("Step 3: Extracting Features")
    if 'feature_list' not in st.session_state:
        st.write("Extracting features from images...")
        feature_list, image_names = extract_features_from_folder(output_dir)
        st.session_state.feature_list = feature_list
        st.session_state.image_names = image_names
        st.success("Features extracted successfully!")
    else:
        st.write("Features are already extracted.")

    # Step 4: Upload query image to find similar images
    st.subheader("Step 4: Upload Query Image")
    uploaded_query_image = st.file_uploader("Upload an image to find similar images", type=["png", "jpg", "jpeg"])

    if uploaded_query_image is not None:
        # Save the uploaded file to a temporary location
        query_image_path = os.path.join("temp_query_image.jpg")
        with open(query_image_path, "wb") as f:
            f.write(uploaded_query_image.getbuffer())
        
        # Display the uploaded image
        st.image(Image.open(query_image_path), caption="Uploaded Image", width=300)
        
        # Find top 5 similar images
        st.write("Finding similar images...")
        top_similar_images = find_top_similar_images(query_image_path, st.session_state.feature_list, st.session_state.image_names, output_dir)
        
        # Display the results
        st.write("Top 5 Similar Images:")

        # Create two columns for better layout
        col1, col2 = st.columns(2)

        # Loop through and display images
        for i, image_path in enumerate(top_similar_images):
            col = col1 if i % 2 == 0 else col2
            col.image(image_path, use_column_width=True, width=250)  # Adjust width here

if __name__ == "__main__":
    main()
