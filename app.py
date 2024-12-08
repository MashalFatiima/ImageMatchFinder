import os
import requests
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the pre-trained VGG-16 model
model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Directories for images and features
image_dir = "downloaded_images"
feature_dir = "features"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(feature_dir, exist_ok=True)

# Function to download images
def download_images(csv_file):
    st.info("Downloading images...")
    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        image_url = row["Image Link"]
        image_id = row["Product ID"]
        output_path = os.path.join(image_dir, f"{image_id}.jpg")
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            with open(output_path, "wb") as file:
                file.write(response.content)
            print(f"Downloaded: {output_path}")
        except Exception as e:
            print(f"Error downloading {image_url}: {e}")
    st.success("Images downloaded successfully!")

# Function to extract features
def extract_features(image_path, model):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array)
        print(f"Extracted features for: {image_path}")
        return features.flatten()
    except Exception as e:
        print(f"Error extracting features for {image_path}: {e}")
        return None

# Function to find top similar images
def find_top_similar_images(query_image_path, model, features_list, image_paths, top_n=5):
    query_vector = extract_features(query_image_path, model)
    if query_vector is None:
        return []
    similarities = []
    for i, stored_vector in enumerate(features_list):
        sim_score = cosine_similarity([query_vector], [stored_vector])[0][0]
        similarities.append((image_paths[i], sim_score))
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

# Streamlit App
st.title("🖼️ Image Similarity Finder")

uploaded_image = st.file_uploader("Upload a query image", type=["jpg", "png"])
if uploaded_image is not None:
    # Save query image
    query_image_path = os.path.join(image_dir, "query_image.jpg")
    with open(query_image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    st.image(query_image_path, caption="Query Image")

    # Check or download dataset
    if not os.listdir(image_dir):
        csv_path = st.text_input("Enter the path to your CSV file with image links:")
        if csv_path:
            download_images(csv_path)

    # Feature extraction
    features_list_path = os.path.join(feature_dir, "features_list.npy")
    image_paths_path = os.path.join(feature_dir, "image_paths.npy")

    if not os.path.exists(features_list_path) or not os.path.exists(image_paths_path):
        st.info("Extracting features...")
        features_list = []
        image_paths = []
        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            feature_vector = extract_features(image_path, model)
            if feature_vector is not None:
                np.save(os.path.join(feature_dir, f"{image_name}.npy"), feature_vector)
                features_list.append(feature_vector)
                image_paths.append(image_path)
        np.save(features_list_path, features_list)
        np.save(image_paths_path, image_paths)
        st.success("Feature extraction completed!")
    else:
        features_list = np.load(features_list_path, allow_pickle=True)
        image_paths = np.load(image_paths_path, allow_pickle=True)

    # Find similar images
    st.info("Finding similar images...")
    top_similar_images = find_top_similar_images(query_image_path, model, features_list, image_paths, top_n=5)

    if top_similar_images:
        st.success("Top 5 similar images found!")
        for idx, (similar_image_path, score) in enumerate(top_similar_images):
            print(f"Rank {idx+1}: {similar_image_path}, Similarity Score: {score}")
            st.image(similar_image_path, caption=f"Rank {idx+1}: Similarity {score:.4f}")
    else:
        st.error("No similar images found. Make sure your dataset contains valid images.")
