# app.py
import pandas as pd
import cv2
import random
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
import io
from starlette.responses import StreamingResponse
import os

# Initialize FastAPI app
app = FastAPI()

# Load the dataset (assuming 'data_mask.csv' is in the same directory)
brain_df = pd.read_csv('../api/data_mask.csv')

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Brain MRI API. Use /image/{index} to get an MRI image and mask."}

# Endpoint to serve a random MRI image and mask or a specific one by index
@app.get("/image/{index}")
def get_image(index: int = None):
    if index is None or index < 0 or index >= len(brain_df):
        # Select a random index if none provided or out of bounds
        index = random.randint(0, len(brain_df) - 1)
    
    # Get image and mask paths
    image_path = brain_df['image_path'][index]
    mask_path = brain_df['mask_path'][index]
    mask_value = brain_df['mask'][index]

    # Read the images using OpenCV
    mri_image = cv2.imread(image_path)
    mask_image = cv2.imread(mask_path)

    # Create a figure with two subplots (MRI and Mask)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(cv2.cvtColor(mri_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("MRI del cerebro")
    axs[0].axis('off')
    axs[1].imshow(cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f"Máscara - {mask_value}")
    axs[1].axis('off')

    # Save the plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)  # Close the figure to free memory

    # Return the image as a streaming response
    return StreamingResponse(buf, media_type="image/png")

# Endpoint to get a specific image file directly (e.g., raw MRI)
@app.get("/raw_image/{index}")
def get_raw_image(index: int):
    if index < 0 or index >= len(brain_df):
        return {"error": "Index out of bounds"}
    image_path = brain_df['image_path'][index]
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/jpeg")
    return {"error": "Image not found"}