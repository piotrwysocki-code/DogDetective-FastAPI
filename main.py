from fastapi import FastAPI, File, UploadFile
import numpy as np
import os
import cv2
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import keras.applications.inception_v3
import keras.applications.xception
import keras.applications.inception_resnet_v2
import keras.applications.nasnet
from keras.models import Model, Sequential, load_model
from keras import layers
from keras import utils as np_utils
from keras.preprocessing.image import ImageDataGenerator
from array import *
from keras.models import load_model

features = load_model("120breedfeatures.h5")
model = load_model("120breedmodel.h5")
n = 331
dogBreeds = []
categoryDict = {}
results = []

app = FastAPI()

with open('categoryDict.pkl', 'rb') as handle:
    categoryDict = pickle.load(handle)

for i in range(0, 120):
    dogBreeds.append(i)
    results.append(categoryDict[i])

@app.get("/")
def index():
    return "Dog Detective FastAPI Server"

@app.get("/breeds")
def index():
    return categoryDict

@app.get("/breeds/{id}")
def index(id: int):
    if(id in dogBreeds):
        return categoryDict[id]
    else:
        return "Breed Not Found"

@app.post("/api/classify")
async def classify(file: UploadFile):
    try:
        contents = await file.read()
        filePath = 'temp/uploads/' + file.filename
        with open(filePath, 'wb') as f:
            f.write(contents)
    except Exception as e:
        print(e)
        if(os.path.exists(filePath)):
            os.remove(filePath)
    finally:
        await file.close()

        result = predict_breed(filePath)

        if(os.path.exists(filePath)):
            os.remove(filePath)

    return result


def feature_extractor(dataframe):
    img_size = (n, n, 3)
    data_size = (len(dataframe))
    batch_size = 1

    x = np.zeros([data_size, 9664], dtype=np.uint8)
    datagen = ImageDataGenerator()
    temp = datagen.flow_from_dataframe(dataframe,
                                       x_col='filename', class_mode=None,
                                       batch_size=1, shuffle=False, target_size=(img_size[:2]), color_mode='rgb')
    i = 0
    for input_batch in temp:
        input_batch = features.predict(input_batch)
        x[i * batch_size: (i + 1) * batch_size] = input_batch
        i += 1
        if i * batch_size >= data_size:
            break
    return x


def predict_breed(test_img_dir):
    test_img_uri = np.asarray(test_img_dir)

    imDF = pd.DataFrame({
        'filename': test_img_uri
    }, index=[0])

    x_imgFeatures = feature_extractor(imDF)
    y_pred = model.predict(x_imgFeatures)

    resultsDF = pd.DataFrame({
        'Breed': results,
        'Confidence': y_pred[0]
    })

    column = resultsDF["Confidence"]
    temp = resultsDF.nlargest(3, "Confidence")

    return temp.to_json(orient='records')
