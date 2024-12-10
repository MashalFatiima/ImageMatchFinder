# Image Match Finder

Image Match Finder is a web application developed using **Streamlit**. The app allows users to upload an image and find the top 5 most similar images from a pre-uploaded dataset using feature extraction and cosine similarity. The feature extraction is done using the **VGG16** model, a deep learning model for image recognition.

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

## Running the App

To run the app locally, navigate to your project directory and run the following command:

```bash
streamlit run app.py
```
This will open a web page in your browser where you can interact with the app.

## Developed By

**Mashal Fatima**
