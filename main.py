from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL = tf.keras.models.load_model("saved_models/1")


class_names = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "hello, i am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    pass
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
