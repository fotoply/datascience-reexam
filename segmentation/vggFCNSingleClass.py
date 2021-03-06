import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.layers import Conv2D, UpSampling2D, Concatenate
import numpy
from PIL import Image, ImageOps
from keras import applications
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array

from segmentation.augmenter import augmentImage, padImage, createMaskFromTransparency

IMG_HEIGHT = 256
IMG_WIDTH = 256


def train():
    print("Preparing models")
    model = defineModel()
    print("Models prepared")
    print("-------------------------------------------")
    print("Loading training data")
    dataExpected, dataInput = prepareData("data/train/polyp/", "data/train/raw/")
    print("Training data loaded")
    print("-------------------------------------------")
    print("Training starting")
    modelname = "vgg_fcn_single_2"
    # Save the model after run
    checkpoint = ModelCheckpoint(modelname + ".h5", monitor='val_acc', verbose=1,
                                 save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1)

    # Stop training early if the model stops improving
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

    # Save the epoch data to a CSV file for future inspection
    csvLog = CSVLogger(modelname + ".csv")

    model.fit(x=dataInput, y=dataExpected, batch_size=8, epochs=60, validation_split=0.15,
              callbacks=[checkpoint, early, csvLog])
    print("All training completed")
    print("-------------------------------------------")
    print("Model has been saved")


def defineModel():
    model = applications.vgg16.VGG16(weights="imagenet", include_top=False,
                                     input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    # Freeze everything in the resnet model, only training classification layers which are added afterwards
    for layer in model.layers:
        layer.trainable = False

    # Adding own layers
    x = model.output
    x = UpSampling2D()(x)
    x = Conv2D(128, 3, padding="same", activation="relu")(x)
    x = Conv2D(128, 3, padding="same", activation="relu")(x)
    x = UpSampling2D()(x)

    identity = model.get_layer("block3_pool").output
    # identity = Conv2D(64, 3, padding="same", activation="relu") (identity)
    x = Concatenate()([x, identity])
    x = Conv2D(64, 3, padding="same", activation="relu")(x)
    x = Conv2D(64, 3, padding="same", activation="relu")(x)
    x = UpSampling2D()(x)

    identity = model.get_layer("block2_pool").output
    # identity = Conv2D(64, 3, padding="same", activation="relu") (identity)
    x = Concatenate()([x, identity])
    x = Conv2D(64, 3, padding="same", activation="relu")(x)
    x = Conv2D(32, 3, padding="same", activation="relu")(x)
    x = UpSampling2D()(x)

    identity = model.get_layer("block1_pool").output
    # identity = Conv2D(64, 3, padding="same", activation="relu") (identity)
    x = Concatenate()([x, identity])
    x = Conv2D(32, 3, padding="same", activation="relu")(x)
    x = Conv2D(32, 3, padding="same", activation="relu")(x)
    x = UpSampling2D()(x)

    identity = model.get_layer("block1_conv2").output
    #identity = Conv2D(64, 3, padding="same", activation="relu") (identity)
    x = Concatenate()([x, identity])
    x = Conv2D(16, 3, padding="same", activation="relu")(x)
    x = Conv2D(8, 3, padding="same", activation="relu")(x)
    prediction = Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    # prediction = Conv2D(3, 1, padding="same",activation="softmax")(x)

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
        testMask = Image.open(images_mask + file.split(".")[0] + ".png")

        for augmented in augmentImage(testImage, "", applyGrayscale=False, printGeneratedLists=False):
            image = augmented[1]
            image = padImage(image, (IMG_WIDTH, IMG_HEIGHT))
            inputValue = img_to_array(image)
            inputValue /= 255
            inputArray.append(inputValue)

        for augmentedMask in augmentImage(createMaskFromTransparency(testMask), "", applyGrayscale=False, isMask=True,
                                          printGeneratedLists=False):
            image = padImage(augmentedMask[1], (IMG_WIDTH, IMG_HEIGHT))
            image = ImageOps.grayscale(image)
            expectedOutput = img_to_array(image)
            expectedOutput /= 255
            expectedArray.append(expectedOutput)
    inputArray = numpy.array(inputArray)
    expectedArray = numpy.array(expectedArray)
    return expectedArray, inputArray


train()
