import os


os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, CSVLogger

img_width, img_height = 227, 227
train_data_dir = "data/train"
validation_data_dir = "data/validation"
train_samples = 250
validation_samples = 60
batch_size = 8
epochs = 50
modelname = "resnet_1"

model = applications.resnet50.ResNet50(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

# Freeze everything in the resnet model, only training classification layers which are added afterwards
for layer in model.layers:
    layer.trainable = False

# Adding own layers
x = model.output
x = Conv2D(16, 3, padding="same", activation="relu")(x)
x = Flatten()(x)
#x = Dense(16, activation="relu")(x)
#x = Dropout(0.8)(x)
x = Dense(8, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model
model_final = Model(inputs=[model.input], outputs=[predictions])
model_final.summary()
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(lr=0.0001),
                    metrics=["accuracy"])

# Initiate the train and test generators with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=180)

validation_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=180)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical")

label_map = train_generator.class_indices
print(label_map)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    class_mode="categorical")

# Save the model after run
checkpoint = ModelCheckpoint(modelname + ".h5", monitor='val_acc', verbose=1,
                             save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)

# Stop training early if the model stops improving
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# Save the epoch data to a CSV file for future inspection
csvLog = CSVLogger(modelname + ".csv")

# Train the model
model_final.fit_generator(
    train_generator,
    validation_data=validation_generator,

    validation_steps=validation_samples,
    steps_per_epoch=train_samples,
    epochs=epochs,
    class_weight={0: 0.35, 1: 1},

    callbacks=[checkpoint, early, csvLog])
