import os
import numpy as np
from keras import Model
from keras.engine.saving import load_model
from keras.preprocessing import image

PATH = os.getcwd()
modelPath = "convnet_1.h5"
labels = {1: "dirty", 0: "clean"}

model = load_model(modelPath)


def predict(imagePath):
    imageList = os.listdir(imagePath)

    for imageName in imageList:
        img_path = imagePath + imageName
        x = image.img_to_array(image.load_img(img_path))
        x = np.expand_dims(x, axis=0)
        predict = model.predict(x)
        predict = predict.argmax(axis=-1)
        print(labels[predict[0]])


clean_path = PATH + '/data/test/clean/'
dirty_path = PATH + '/data/test/dirty/'
predict(clean_path)
predict(dirty_path)
