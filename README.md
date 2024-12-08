# Image Similarity Finder

This project is an **Image Similarity Finder** web application built using **Streamlit** and **TensorFlow**. The app allows users to upload a query image and find the top 5 most similar images from a dataset. The similarity between images is calculated using **cosine similarity** based on the feature vectors extracted from the pre-trained **VGG-16** deep learning model.

## Features

- **Query Image Upload**: Users can upload a query image to find similar images.
- **Dataset Download from CSV**: Automatically downloads dataset images from a CSV file containing image URLs.
- **Cosine Similarity**: The similarity between the uploaded query image and dataset images is calculated using cosine similarity.
- **Top 5 Similar Images**: Displays the top 5 most similar images to the query image.
- **Dynamic Processing**: When a new query image is uploaded, the results are reset, and the app starts processing from scratch.
- **Visual Display**: The query image and the top 5 similar images are displayed in a visually appealing format using **Streamlit**.

## Dependencies

- Python 3.x
- `streamlit` library
- `numpy` library
- `pandas` library
- `tensorflow` library
- `scikit-learn` library
- `requests` library
- `Pillow` library

## Code Overview

### Main Components

- **`download_images(csv_file)`**: Downloads images from the URLs provided in the CSV file.
- **`extract_features(image_path, model)`**: Extracts feature vectors from images using the pre-trained VGG-16 model.
- **`find_top_similar_images(query_image_path, model, features_list, image_paths, top_n=5)`**: Finds the top N similar images by calculating cosine similarity.
- **`streamlit` Interface**: The app uses Streamlit to provide an interactive web interface, allowing users to upload a query image and view the results.

### App Flow

1. **Image Upload**: The user uploads a query image via the Streamlit interface.
2. **Image Download**: If dataset images are not already present, the app will download them from the CSV file provided by the user.
3. **Feature Extraction**: The app extracts feature vectors from both the query image and the dataset images.
4. **Similarity Calculation**: The app calculates cosine similarity between the query image and all dataset images.
5. **Display Results**: The top 5 most similar images are displayed with their similarity scores.

## Controls

- **Upload Image**: Click on the "Upload a query image" button to upload your query image in **JPG/PNG** format.
- **CSV File Path**: Enter the path to your CSV file containing the image links if dataset images need to be downloaded.
- **Reset Results**: Upload a new image to reset the process and start with a fresh query.

## How to Run

1. Make sure you have Python installed on your system.
2. Install the required libraries using pip:
   ```bash
   pip install streamlit numpy pandas tensorflow scikit-learn pillow requests
3. Save the code in a file named app.py.
4. Run the Streamlit app:
   ```bash
   streamlit run path\app.py

## Example Flow
1. Upload a Query Image: You start by uploading a query image to the app.
2. CSV Input: You enter the path to the CSV file that contains the image URLs. The app will download the dataset images if not already present.
3. Get Results: The app processes the images, calculates cosine similarities, and displays the top 5 most similar images.
