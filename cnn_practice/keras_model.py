# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = np.reshape(x_train, (60000, 28, 28, 1))
x_test = np.reshape(x_test, (10000, 28, 28, 1))

model = tf.keras.models.Sequential([
    layers.Conv2D(64, (2, 2), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30)
model.evaluate(x_test, y_test)

# test accuracy: 0.8942
