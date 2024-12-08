import os
import requests
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

image_dir = "downloaded_images"
feature_dir = "features"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(feature_dir, exist_ok=True)

def download_images(csv_file):
    st.info("Downloading images from the dataset...")
    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        image_url = row["image_link"]
        image_id = row["Product ID"]
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            output_path = os.path.join(image_dir, f"{image_id}.jpg")
            with open(output_path, "wb") as file:
                file.write(response.content)
    st.success(f"All images downloaded successfully to {image_dir}!")

def extract_features(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

def find_top_similar_images(query_image_path, model, features_list, image_paths, top_n=5):
    query_vector = extract_features(query_image_path, model)
    similarities = []
    for i, stored_vector in enumerate(features_list):
        sim_score = cosine_similarity([query_vector], [stored_vector])[0][0]
        similarities.append((image_paths[i], sim_score))
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

st.title("🖼️ Image Similarity Finder")
st.markdown(
    """
    **Step 1:** Upload a query image to start.  
    **Step 2:** The app will automatically download images from the provided dataset CSV and process them for similarity matching.  
    **Step 3:** View the top 5 similar images based on cosine similarity.
    """
)

uploaded_image = st.file_uploader("Upload a query image to find similar images", type=["jpg", "png"])

if uploaded_image is not None:
    query_image_path = os.path.join(image_dir, "query_image.jpg")
    with open(query_image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    st.image(query_image_path, caption="Query Image", use_column_width=True)

    if not os.listdir(image_dir):
        csv_path = st.text_input("Enter the path to your CSV file with image links:")
        if csv_path:
            download_images(csv_path)

    features_list_path = os.path.join(feature_dir, "features_list_all.npy")
    image_paths_path = os.path.join(feature_dir, "image_paths_all.npy")

    if not os.path.exists(features_list_path) or not os.path.exists(image_paths_path):
        st.info("Extracting features for the dataset...")
        features_list = []
        image_paths = []
        for image_name in os.listdir(image_dir):
            if image_name.endswith(('.jpg', '.png')) and image_name != "query_image.jpg":
                image_path = os.path.join(image_dir, image_name)
                feature_vector = extract_features(image_path, model)
                np.save(os.path.join(feature_dir, f"{image_name}.npy"), feature_vector)
                features_list.append(feature_vector)
                image_paths.append(image_path)
        np.save(features_list_path, features_list)
        np.save(image_paths_path, image_paths)
        st.success("Feature extraction completed!")
    else:
        features_list = np.load(features_list_path, allow_pickle=True)
        image_paths = np.load(image_paths_path, allow_pickle=True)

    st.info("Finding similar images...")
    top_similar_images = find_top_similar_images(query_image_path, model, features_list, image_paths, top_n=5)

    st.success("Top 5 similar images found!")
    st.write("### Similar Images:")
    for idx, (similar_image_path, score) in enumerate(top_similar_images):
        st.image(similar_image_path, caption=f"Rank {idx+1}: Similarity {score:.4f}", use_column_width=True)

    st.warning("Upload a new query image to reset and find different results.")
