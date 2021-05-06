

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #disable Tensorflow warning messages

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()


# convert class vectors to binary class matrices
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(16, activation="sigmoid"),
        layers.Dense(16, activation="sigmoid"),
        layers.Dense(num_classes, activation="sigmoid"),
    ]
)
model.summary()


batch_size = 50
epochs = 20

model.compile(loss="MSE", optimizer="adam", metrics=["accuracy"])
model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)

model.save("3b1bModel.hdf5")

score = model.evaluate(test_images, test_labels, verbose=0)
print(model.metrics_names)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
