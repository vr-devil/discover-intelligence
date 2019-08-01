# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = np.reshape(x_train, (60000, 28, 28, 1))
x_test = np.reshape(x_test, (10000, 28, 28, 1))

# x_train, y_train = x_train[:10000], y_train[:10000]
# x_val, y_val = x_test[:1000], y_test[:1000]

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
# y_val = tf.keras.utils.to_categorical(y_val, 10)

model = tf.keras.models.Sequential([
    layers.Conv2D(55, (2, 2), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPool2D((2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(190, (2, 2), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.BatchNormalization(),
    layers.Dense(2048, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(2048, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1000, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

# Train on 60000 samples, validate on 10000 samples
# Epoch 1/50
# 60000/60000 [==============================] - 42s 697us/sample - loss: 0.3672 - acc: 0.8727 - val_loss: 0.2626 - val_acc: 0.9056
# Epoch 2/50
# 60000/60000 [==============================] - 39s 654us/sample - loss: 0.2044 - acc: 0.9260 - val_loss: 0.2539 - val_acc: 0.9075
# Epoch 3/50
# 60000/60000 [==============================] - 40s 661us/sample - loss: 0.1445 - acc: 0.9475 - val_loss: 0.2425 - val_acc: 0.9164
# Epoch 4/50
# 60000/60000 [==============================] - 39s 654us/sample - loss: 0.1055 - acc: 0.9614 - val_loss: 0.2508 - val_acc: 0.9172
# Epoch 5/50
# 60000/60000 [==============================] - 40s 664us/sample - loss: 0.0767 - acc: 0.9741 - val_loss: 0.2688 - val_acc: 0.9147
# Epoch 6/50
# 60000/60000 [==============================] - 40s 660us/sample - loss: 0.0547 - acc: 0.9822 - val_loss: 0.2569 - val_acc: 0.9193
# Epoch 7/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0418 - acc: 0.9872 - val_loss: 0.2819 - val_acc: 0.9178
# Epoch 8/50
# 60000/60000 [==============================] - 39s 656us/sample - loss: 0.0309 - acc: 0.9913 - val_loss: 0.2854 - val_acc: 0.9198
# Epoch 9/50
# 60000/60000 [==============================] - 40s 672us/sample - loss: 0.0243 - acc: 0.9935 - val_loss: 0.2898 - val_acc: 0.9206
# Epoch 10/50
# 60000/60000 [==============================] - 39s 656us/sample - loss: 0.0196 - acc: 0.9955 - val_loss: 0.2983 - val_acc: 0.9208
# Epoch 11/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0161 - acc: 0.9962 - val_loss: 0.3011 - val_acc: 0.9233
# Epoch 12/50
# 60000/60000 [==============================] - 39s 654us/sample - loss: 0.0145 - acc: 0.9966 - val_loss: 0.3105 - val_acc: 0.9213
# Epoch 13/50
# 60000/60000 [==============================] - 39s 654us/sample - loss: 0.0113 - acc: 0.9980 - val_loss: 0.3123 - val_acc: 0.9212
# Epoch 14/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0098 - acc: 0.9982 - val_loss: 0.3159 - val_acc: 0.9224
# Epoch 15/50
# 60000/60000 [==============================] - 39s 654us/sample - loss: 0.0094 - acc: 0.9979 - val_loss: 0.3129 - val_acc: 0.9249
# Epoch 16/50
# 60000/60000 [==============================] - 39s 655us/sample - loss: 0.0071 - acc: 0.9988 - val_loss: 0.3176 - val_acc: 0.9232
# Epoch 17/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0072 - acc: 0.9987 - val_loss: 0.3232 - val_acc: 0.9246
# Epoch 18/50
# 60000/60000 [==============================] - 39s 654us/sample - loss: 0.0058 - acc: 0.9992 - val_loss: 0.3333 - val_acc: 0.9218
# Epoch 19/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0051 - acc: 0.9993 - val_loss: 0.3312 - val_acc: 0.9243
# Epoch 20/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0049 - acc: 0.9993 - val_loss: 0.3395 - val_acc: 0.9217
# Epoch 21/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0047 - acc: 0.9994 - val_loss: 0.3379 - val_acc: 0.9239
# Epoch 22/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0039 - acc: 0.9995 - val_loss: 0.3501 - val_acc: 0.9228
# Epoch 23/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0036 - acc: 0.9996 - val_loss: 0.3416 - val_acc: 0.9249
# Epoch 24/50
# 60000/60000 [==============================] - 39s 654us/sample - loss: 0.0039 - acc: 0.9994 - val_loss: 0.3463 - val_acc: 0.9232
# Epoch 25/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0037 - acc: 0.9993 - val_loss: 0.3521 - val_acc: 0.9219
# Epoch 26/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0032 - acc: 0.9997 - val_loss: 0.3500 - val_acc: 0.9237
# Epoch 27/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0034 - acc: 0.9995 - val_loss: 0.3470 - val_acc: 0.9249
# Epoch 28/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0030 - acc: 0.9997 - val_loss: 0.3467 - val_acc: 0.9245
# Epoch 29/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0030 - acc: 0.9996 - val_loss: 0.3564 - val_acc: 0.9238
# Epoch 30/50
# 60000/60000 [==============================] - 39s 654us/sample - loss: 0.0027 - acc: 0.9996 - val_loss: 0.3558 - val_acc: 0.9236
# Epoch 31/50
# 60000/60000 [==============================] - 39s 655us/sample - loss: 0.0025 - acc: 0.9997 - val_loss: 0.3567 - val_acc: 0.9245
# Epoch 32/50
# 60000/60000 [==============================] - 39s 655us/sample - loss: 0.0022 - acc: 0.9998 - val_loss: 0.3561 - val_acc: 0.9235
# Epoch 33/50
# 60000/60000 [==============================] - 39s 654us/sample - loss: 0.0021 - acc: 0.9998 - val_loss: 0.3583 - val_acc: 0.9239
# Epoch 34/50
# 60000/60000 [==============================] - 39s 654us/sample - loss: 0.0019 - acc: 0.9999 - val_loss: 0.3582 - val_acc: 0.9246
# Epoch 35/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0018 - acc: 0.9998 - val_loss: 0.3647 - val_acc: 0.9249
# Epoch 36/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0018 - acc: 0.9999 - val_loss: 0.3640 - val_acc: 0.9249
# Epoch 37/50
# 60000/60000 [==============================] - 39s 652us/sample - loss: 0.0018 - acc: 0.9999 - val_loss: 0.3679 - val_acc: 0.9235
# Epoch 38/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0023 - acc: 0.9996 - val_loss: 0.3737 - val_acc: 0.9243
# Epoch 39/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0020 - acc: 0.9998 - val_loss: 0.3695 - val_acc: 0.9225
# Epoch 40/50
# 60000/60000 [==============================] - 39s 655us/sample - loss: 0.0017 - acc: 0.9998 - val_loss: 0.3731 - val_acc: 0.9241
# Epoch 41/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0016 - acc: 0.9998 - val_loss: 0.3714 - val_acc: 0.9252
# Epoch 42/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0014 - acc: 0.9999 - val_loss: 0.3731 - val_acc: 0.9259
# Epoch 43/50
# 60000/60000 [==============================] - 39s 654us/sample - loss: 0.0014 - acc: 0.9999 - val_loss: 0.3687 - val_acc: 0.9244
# Epoch 44/50
# 60000/60000 [==============================] - 39s 654us/sample - loss: 0.0014 - acc: 0.9999 - val_loss: 0.3746 - val_acc: 0.9244
# Epoch 45/50
# 60000/60000 [==============================] - 39s 654us/sample - loss: 0.0015 - acc: 0.9998 - val_loss: 0.3749 - val_acc: 0.9255
# Epoch 46/50
# 60000/60000 [==============================] - 39s 652us/sample - loss: 0.0015 - acc: 0.9998 - val_loss: 0.3744 - val_acc: 0.9248
# Epoch 47/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0013 - acc: 0.9999 - val_loss: 0.3727 - val_acc: 0.9245
# Epoch 48/50
# 60000/60000 [==============================] - 39s 652us/sample - loss: 0.0012 - acc: 0.9999 - val_loss: 0.3810 - val_acc: 0.9228
# Epoch 49/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0013 - acc: 0.9998 - val_loss: 0.3795 - val_acc: 0.9244
# Epoch 50/50
# 60000/60000 [==============================] - 39s 653us/sample - loss: 0.0011 - acc: 0.9999 - val_loss: 0.3753 - val_acc: 0.9261
# 10000/10000 [==============================] - 1s 114us/sample - loss: 0.3753 - acc: 0.9261



