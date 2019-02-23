import os

import numpy as np
from keras.engine.saving import load_model
from keras.preprocessing import image
from keras.preprocessing.image import array_to_img

from segmentation.augmenter import padImage

PATH = os.getcwd()
modelPath = "vgg_fcn_single_2.h5"
model = load_model(modelPath)

def predict(imagePath):
    imageList = os.listdir(imagePath)

    for imageName in imageList:
        img_path = imagePath + imageName

        pad_image = padImage(image.load_img(img_path), (256, 256))
        x = image.img_to_array(pad_image)
        x /= 255
        x = np.expand_dims(x, axis=0)
        predict = model.predict(x)
        predict[0] *= 255
        array_to_img(predict[0]).show()


path = PATH + '/data/test/raw/'
predict(path)
input()