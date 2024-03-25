from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
import matplotlib.pyplot as plt  # Bibliothèque pour la visualisation de données
import keras  # Bibliothèque pour la création de modèles de réseaux de neurones
from PIL import Image
import numpy as np
import cv2

app = FastAPI()

model = load_model('mnist.model')

IMG_SIZE = 28


@app.get("/")
def FirstAPI():
    return "Hello World"


@app.get("/check")
def check_model_import():
    return model.summary()


@app.post("/predict")
# predict a image passed in the request by post method
async def predict_image(file: UploadFile):
    # Lire et traiter l'image
    img = Image.open(file.file)
    if img.mode == 'RGB':
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    else:
        gray = np.array(img)

    # Redimensionner l'image si nécessaire
    if gray.shape != (IMG_SIZE, IMG_SIZE):
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_AREA)

    # Normaliser l'image
    norm_img = keras.utils.normalize(gray, axis=1)
    norm_img = np.array(norm_img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Faire la prédiction
    prediction = model.predict(norm_img)
    result = np.argmax(prediction)

    return {"prediction": int(result)}
