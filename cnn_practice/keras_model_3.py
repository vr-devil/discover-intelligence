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
    layers.Dense(1000, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

# Train on 60000 samples, validate on 10000 samples
# Epoch 1/50
# 60000/60000 [==============================] - 35s 585us/sample - loss: 0.3557 - acc: 0.8759 - val_loss: 0.2814 - val_acc: 0.8963
# Epoch 2/50
# 60000/60000 [==============================] - 33s 551us/sample - loss: 0.2049 - acc: 0.9263 - val_loss: 0.2423 - val_acc: 0.9160
# Epoch 3/50
# 60000/60000 [==============================] - 35s 577us/sample - loss: 0.1532 - acc: 0.9452 - val_loss: 0.2413 - val_acc: 0.9149
# Epoch 4/50
# 60000/60000 [==============================] - 35s 579us/sample - loss: 0.1179 - acc: 0.9585 - val_loss: 0.2448 - val_acc: 0.9158
# Epoch 5/50
# 60000/60000 [==============================] - 34s 563us/sample - loss: 0.0914 - acc: 0.9697 - val_loss: 0.2485 - val_acc: 0.9152
# Epoch 6/50
# 60000/60000 [==============================] - 34s 563us/sample - loss: 0.0716 - acc: 0.9771 - val_loss: 0.2445 - val_acc: 0.9201
# Epoch 7/50
# 60000/60000 [==============================] - 33s 555us/sample - loss: 0.0570 - acc: 0.9827 - val_loss: 0.2486 - val_acc: 0.9197
# Epoch 8/50
# 60000/60000 [==============================] - 33s 557us/sample - loss: 0.0467 - acc: 0.9867 - val_loss: 0.2537 - val_acc: 0.9198
# Epoch 9/50
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0363 - acc: 0.9909 - val_loss: 0.2576 - val_acc: 0.9201
# Epoch 10/50
# 60000/60000 [==============================] - 33s 549us/sample - loss: 0.0304 - acc: 0.9926 - val_loss: 0.2605 - val_acc: 0.9196
# Epoch 11/50
# 60000/60000 [==============================] - 33s 545us/sample - loss: 0.0260 - acc: 0.9940 - val_loss: 0.2632 - val_acc: 0.9230
# Epoch 12/50
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0229 - acc: 0.9954 - val_loss: 0.2681 - val_acc: 0.9206
# Epoch 13/50
# 60000/60000 [==============================] - 33s 547us/sample - loss: 0.0180 - acc: 0.9969 - val_loss: 0.2775 - val_acc: 0.9197
# Epoch 14/50
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0163 - acc: 0.9974 - val_loss: 0.2787 - val_acc: 0.9195
# Epoch 15/50
# 60000/60000 [==============================] - 33s 548us/sample - loss: 0.0142 - acc: 0.9979 - val_loss: 0.2800 - val_acc: 0.9214
# Epoch 16/50
# 60000/60000 [==============================] - 34s 559us/sample - loss: 0.0130 - acc: 0.9980 - val_loss: 0.2849 - val_acc: 0.9215
# Epoch 17/50
# 60000/60000 [==============================] - 33s 547us/sample - loss: 0.0113 - acc: 0.9985 - val_loss: 0.2851 - val_acc: 0.9230
# Epoch 18/50
# 60000/60000 [==============================] - 33s 547us/sample - loss: 0.0103 - acc: 0.9987 - val_loss: 0.2893 - val_acc: 0.9207
# Epoch 19/50
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0094 - acc: 0.9988 - val_loss: 0.2918 - val_acc: 0.9210
# Epoch 20/50
# 60000/60000 [==============================] - 33s 552us/sample - loss: 0.0084 - acc: 0.9991 - val_loss: 0.2954 - val_acc: 0.9215
# Epoch 21/50
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0081 - acc: 0.9991 - val_loss: 0.2958 - val_acc: 0.9220
# Epoch 22/50
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0077 - acc: 0.9991 - val_loss: 0.2994 - val_acc: 0.9231
# Epoch 23/50
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0070 - acc: 0.9993 - val_loss: 0.2995 - val_acc: 0.9220
# Epoch 24/50
# 60000/60000 [==============================] - 33s 556us/sample - loss: 0.0067 - acc: 0.9991 - val_loss: 0.2992 - val_acc: 0.9232
# Epoch 25/50
# 60000/60000 [==============================] - 34s 566us/sample - loss: 0.0059 - acc: 0.9996 - val_loss: 0.3036 - val_acc: 0.9220
# Epoch 26/50
# 60000/60000 [==============================] - 33s 553us/sample - loss: 0.0057 - acc: 0.9995 - val_loss: 0.3039 - val_acc: 0.9220
# Epoch 27/50
# 60000/60000 [==============================] - 33s 556us/sample - loss: 0.0049 - acc: 0.9997 - val_loss: 0.3097 - val_acc: 0.9234
# Epoch 28/50
# 60000/60000 [==============================] - 33s 548us/sample - loss: 0.0049 - acc: 0.9998 - val_loss: 0.3054 - val_acc: 0.9233
# Epoch 29/50
# 60000/60000 [==============================] - 33s 555us/sample - loss: 0.0048 - acc: 0.9996 - val_loss: 0.3095 - val_acc: 0.9229
# Epoch 30/50
# 60000/60000 [==============================] - 33s 555us/sample - loss: 0.0046 - acc: 0.9997 - val_loss: 0.3091 - val_acc: 0.9235
# Epoch 31/50
# 60000/60000 [==============================] - 34s 565us/sample - loss: 0.0042 - acc: 0.9997 - val_loss: 0.3114 - val_acc: 0.9235
# Epoch 32/50
# 60000/60000 [==============================] - 34s 561us/sample - loss: 0.0041 - acc: 0.9998 - val_loss: 0.3112 - val_acc: 0.9230
# Epoch 33/50
# 60000/60000 [==============================] - 33s 555us/sample - loss: 0.0041 - acc: 0.9996 - val_loss: 0.3138 - val_acc: 0.9227
# Epoch 34/50
# 60000/60000 [==============================] - 33s 547us/sample - loss: 0.0036 - acc: 0.9998 - val_loss: 0.3152 - val_acc: 0.9222
# Epoch 35/50
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0037 - acc: 0.9997 - val_loss: 0.3222 - val_acc: 0.9224
# Epoch 36/50
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0033 - acc: 0.9999 - val_loss: 0.3197 - val_acc: 0.9215
# Epoch 37/50
# 60000/60000 [==============================] - 33s 545us/sample - loss: 0.0034 - acc: 0.9998 - val_loss: 0.3183 - val_acc: 0.9236
# Epoch 38/50
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0033 - acc: 0.9997 - val_loss: 0.3191 - val_acc: 0.9236
# Epoch 39/50
# 60000/60000 [==============================] - 33s 545us/sample - loss: 0.0031 - acc: 0.9998 - val_loss: 0.3223 - val_acc: 0.9229
# Epoch 40/50
# 60000/60000 [==============================] - 33s 545us/sample - loss: 0.0030 - acc: 0.9999 - val_loss: 0.3231 - val_acc: 0.9226
# Epoch 41/50
# 60000/60000 [==============================] - 33s 545us/sample - loss: 0.0030 - acc: 0.9998 - val_loss: 0.3235 - val_acc: 0.9221
# Epoch 42/50
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0027 - acc: 0.9999 - val_loss: 0.3230 - val_acc: 0.9238
# Epoch 43/50
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0026 - acc: 0.9999 - val_loss: 0.3247 - val_acc: 0.9224
# Epoch 44/50
# 60000/60000 [==============================] - 33s 548us/sample - loss: 0.0024 - acc: 1.0000 - val_loss: 0.3249 - val_acc: 0.9234
# Epoch 45/50
# 60000/60000 [==============================] - 33s 545us/sample - loss: 0.0025 - acc: 0.9999 - val_loss: 0.3299 - val_acc: 0.9230
# Epoch 46/50
# 60000/60000 [==============================] - 33s 545us/sample - loss: 0.0024 - acc: 0.9999 - val_loss: 0.3270 - val_acc: 0.9225
# Epoch 47/50
# 60000/60000 [==============================] - 34s 561us/sample - loss: 0.0021 - acc: 0.9999 - val_loss: 0.3283 - val_acc: 0.9226
# Epoch 48/50
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0023 - acc: 0.9999 - val_loss: 0.3259 - val_acc: 0.9237
# Epoch 49/50
# 60000/60000 [==============================] - 34s 562us/sample - loss: 0.0020 - acc: 0.9999 - val_loss: 0.3312 - val_acc: 0.9215
# Epoch 50/50
# 60000/60000 [==============================] - 33s 553us/sample - loss: 0.0020 - acc: 0.9999 - val_loss: 0.3302 - val_acc: 0.9236
# 10000/10000 [==============================] - 1s 108us/sample - loss: 0.3302 - acc: 0.9236


