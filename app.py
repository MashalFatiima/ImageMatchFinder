import os
import zipfile
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the pre-trained VGG-16 model
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Directories for images and features
image_dir = "extracted_images"
feature_dir = "features"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(feature_dir, exist_ok=True)

# Function to extract ZIP file
def extract_images(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(image_dir)
    st.success("Images extracted successfully!")

# Function to extract features from an image
def extract_features(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Function to compute similarity and find top matches
def find_top_similar_images(query_image_path, model, features_list, image_paths, top_n=5):
    query_vector = extract_features(query_image_path, model)
    similarities = []
    for i, stored_vector in enumerate(features_list):
        sim_score = cosine_similarity([query_vector], [stored_vector])[0][0]
        similarities.append((image_paths[i], sim_score))
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

# Streamlit App
st.title("🖼️ Image Similarity Finder")
st.markdown(
    """
    **Step 1:** Upload a ZIP file containing your dataset images (only the first time).  
    **Step 2:** Upload your query image to find the top 5 similar images from the dataset.
    """
)

# Step 1: ZIP file uploader
if not os.listdir(image_dir):
    st.subheader("Step 1: Upload your dataset")
    uploaded_zip = st.file_uploader("Upload a ZIP file containing images", type=["zip"])
    if uploaded_zip is not None:
        st.info("Extracting images...")
        extract_images(uploaded_zip)

        # Extract features for the dataset
        st.info("Extracting features from the dataset...")
        features_list = []
        image_paths = []
        for image_name in os.listdir(image_dir):
            if image_name.endswith(('.jpg', '.png')):
                image_path = os.path.join(image_dir, image_name)
                feature_vector = extract_features(image_path, model)
                np.save(os.path.join(feature_dir, f"{image_name}.npy"), feature_vector)
                features_list.append(feature_vector)
                image_paths.append(image_path)
        np.save(os.path.join(feature_dir, "features_list_all.npy"), features_list)
        np.save(os.path.join(feature_dir, "image_paths_all.npy"), image_paths)
        st.success("Dataset processed successfully!")
else:
    # Load preprocessed features and image paths
    features_list = np.load(os.path.join(feature_dir, "features_list_all.npy"), allow_pickle=True)
    image_paths = np.load(os.path.join(feature_dir, "image_paths_all.npy"), allow_pickle=True)

# Step 2: Query image uploader
st.subheader("Step 2: Upload a Query Image")
uploaded_image = st.file_uploader("Upload a query image", type=["jpg", "png"])

if uploaded_image is not None:
    # Save the query image
    query_image_path = os.path.join(image_dir, "query_image.jpg")
    with open(query_image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Display the query image
    st.image(query_image_path, caption="Query Image", use_column_width=True)

    # Find top 5 similar images
    st.info("Finding similar images...")
    top_similar_images = find_top_similar_images(query_image_path, model, features_list, image_paths, top_n=5)

    # Display results
    st.success("Top 5 similar images found!")
    st.write("### Similar Images:")
    for idx, (similar_image_path, score) in enumerate(top_similar_images):
        st.image(similar_image_path, caption=f"Rank {idx+1}: Similarity {score:.4f}", use_column_width=True)

    # Clear results if a new image is uploaded
    st.warning("Upload a new query image to reset and find different results.")
