import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Reshape, UpSampling2D
import numpy
from PIL import Image, ImageOps
from keras import applications
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array

from segmentation.augmenter import augmentImage, padImage

IMG_HEIGHT = 256
IMG_WIDTH = 256


def train():
    print("Preparing models")
    model = defineModel()
    print("Models prepared")
    print("-------------------------------------------")
    print("Loading training data")
    dataExpected, dataInput = prepareData("data/train/mask/", "data/train/raw/")
    print("Training data loaded")
    print("-------------------------------------------")
    print("Training starting")
    modelname = "resnet_fcn"
    # Save the model after run
    checkpoint = ModelCheckpoint(modelname + ".h5", monitor='val_acc', verbose=1,
                                 save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1)

    # Stop training early if the model stops improving
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

    # Save the epoch data to a CSV file for future inspection
    csvLog = CSVLogger(modelname + ".csv")

    model.fit(x=dataInput, y=dataExpected, batch_size=8, epochs=10, validation_split=0.15,
              callbacks=[checkpoint, early, csvLog])
    print("All training completed")
    print("-------------------------------------------")
    print("Model has been saved")


def defineModel():
    model = applications.resnet50.ResNet50(weights="imagenet", include_top=False,
                                           input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    # Freeze everything in the resnet model, only training classification layers which are added afterwards
    for layer in model.layers:
        layer.trainable = False

    # Adding own layers
    x = model.output
    x = UpSampling2D()(x)
    x = Conv2D(256, 3, padding="same", activation="relu")(x)
    x = UpSampling2D()(x)
    x = Conv2D(128, 3, padding="same", activation="relu")(x)
    x = UpSampling2D()(x)
    x = Conv2D(64, 3, padding="same", activation="relu")(x)
    x = UpSampling2D()(x)
    x = Conv2D(32, 3, padding="same", activation="relu")(x)
    x = UpSampling2D()(x)
    x = Conv2D(16, 3, padding="same", activation="relu")(x)
    prediction = Conv2D(3, 1, padding="same",activation="relu")(x)

    model = Model(inputs=model.input, outputs=prediction)
    model.compile(optimizer=Adam(lr=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def prepareData(images_mask, images_raw):
    inputArray = []
    expectedArray = []
    for file in os.listdir(images_raw):
        testImage = Image.open(images_raw + file)
        testMask = Image.open(images_mask + file.split(".")[0] + ".jpg")

        for augmented in augmentImage(testImage, "", applyGrayscale=True, printGeneratedLists=False):
            image = augmented[1]
            image = padImage(image, (IMG_WIDTH, IMG_HEIGHT))
            inputValue = img_to_array(image)
            inputValue /= 255
            inputArray.append(inputValue)

        for augmentedMask in augmentImage(testMask, "", applyGrayscale=True, isMask=True,
                                          printGeneratedLists=False):
            image = padImage(augmentedMask[1], (IMG_WIDTH, IMG_HEIGHT))
            expectedOutput = img_to_array(image)
            expectedOutput /= 255
            expectedArray.append(expectedOutput)
    inputArray = numpy.array(inputArray)
    expectedArray = numpy.array(expectedArray)
    return expectedArray, inputArray


train()
