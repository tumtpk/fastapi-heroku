from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from PIL import Image, ImageOps
import io
import tensorflow as tf
import numpy as np
import pandas as pd
import uvicorn
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware

image_directory = Path("images")

app = FastAPI()

class ImageData(BaseModel):
    image_data: str

class Prediction(BaseModel):
    prediction: str

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tf.keras.models.load_model("models/keras_Model.h5", compile=False)

# Load the labels
class_names = open("models/labels.txt", "r").readlines()

# Load the Excel file (Assuming the file is named 'data.xlsx' and has a sheet named 'Sheet1')
data_file = "data/data.xlsx"
df = pd.read_excel(data_file, sheet_name="Sheet1")

def preprocess_image(base64_string):
    # Convert Base64-encoded string to bytes
    image_bytes = base64.b64decode(base64_string.split(",")[1])

    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize and crop the image to 224x224
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert the image to a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    return normalized_image_array

def predict_image(image_data):
    # Preprocess the image
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = preprocess_image(image_data)

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name[2:], confidence_score

def read_html_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:  # Specify the encoding as utf-8
        content = file.read()
    return content

@app.get("/test")
async def test():
    return {"result": "Object detection prediction result"}


@app.post("/predict/")
async def predict(data: ImageData):
    try:
        image_data = data.image_data
        class_name, confidence_score = predict_image(image_data)
        return {"prediction": class_name, "confidence_score": float(confidence_score)}
    except Exception as e:
        raise HTTPException(status_code=422, detail="Error processing the image")
    
@app.post("/process_prediction")
async def process_prediction(input: Prediction):
    # Filter the data based on the prediction
    filtered_data = df[df['Prediction'] == input.prediction]
    if len(filtered_data) > 0:
        # If there is matching data, take the first record
        result_data = filtered_data.iloc[0].to_dict()
        return result_data
    else:
        # If no matching data, return an empty object
        return {}
    
@app.get("/", response_class=HTMLResponse)
def read_root():
    html_file_path = "views/index.html"
    html_content = read_html_file(html_file_path)
    return html_content

@app.get("/camera", response_class=HTMLResponse)
def read_root():
    html_file_path = "views/camera.html"
    html_content = read_html_file(html_file_path)
    return html_content

@app.get("/result", response_class=HTMLResponse)
def read_root():
    html_file_path = "views/result.html"
    html_content = read_html_file(html_file_path)
    return html_content

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    image_path = image_directory / image_name
    return FileResponse(image_path)

# Set up CORS (Allow requests from all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)