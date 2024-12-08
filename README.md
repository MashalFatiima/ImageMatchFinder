
# Image Match Finder

Image Match Finder is a web application developed by **Mashal Fatima** using **Streamlit**. The app allows users to upload an image and find the top 5 most similar images from a pre-uploaded dataset using feature extraction and cosine similarity. The feature extraction is done using the **VGG16** model, a deep learning model for image recognition.

## Features:
- Upload an image to find similar images in a pre-defined dataset.
- Automatically download images from URLs stored in a CSV file.
- Pre-extract features for images to speed up the similarity search.
- Display the top 5 most similar images based on cosine similarity.

## Requirements

- Python 3.7 or above
- Install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## How to Use

### Step 1: Upload the `DataSheet.csv`
Ensure that the `DataSheet.csv` file is in the same directory as `app.py`. The app will automatically read the image links from the CSV and download the images to the `downloaded_images/` directory.

### Step 2: Image Feature Extraction
Once the images are downloaded, the app will automatically extract features using the **VGG16** model. This feature extraction will only happen once unless you upload new images.

### Step 3: Upload Query Image
After uploading the query image, the app will compare it with the existing dataset and display the top 5 most similar images based on **cosine similarity**.

### Step 4: View Results
The top 5 similar images will be displayed in a grid layout. You can click on any image to view it.

## Running the App

To run the app locally, navigate to your project directory and run the following command:

```bash
streamlit run app.py
```
This will open a web page in your browser where you can interact with the app.

## Developed By

**Mashal Fatima**

---
Feel free to contribute or suggest improvements.
