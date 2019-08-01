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
    layers.Conv2D(96, (4, 4), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.4),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(1024, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

# Train on 60000 samples, validate on 10000 samples
# Epoch 1/50
# 60000/60000 [==============================] - 28s 472us/sample - loss: 0.5474 - acc: 0.8299 - val_loss: 0.3610 - val_acc: 0.8701
# Epoch 2/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.3330 - acc: 0.8783 - val_loss: 0.3108 - val_acc: 0.8866
# Epoch 3/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.2936 - acc: 0.8933 - val_loss: 0.2891 - val_acc: 0.8943
# Epoch 4/50
# 60000/60000 [==============================] - 26s 440us/sample - loss: 0.2696 - acc: 0.9022 - val_loss: 0.2750 - val_acc: 0.9008
# Epoch 5/50
# 60000/60000 [==============================] - 26s 441us/sample - loss: 0.2497 - acc: 0.9088 - val_loss: 0.2719 - val_acc: 0.9026
# Epoch 6/50
# 60000/60000 [==============================] - 26s 440us/sample - loss: 0.2353 - acc: 0.9145 - val_loss: 0.2553 - val_acc: 0.9091
# Epoch 7/50
# 60000/60000 [==============================] - 26s 440us/sample - loss: 0.2220 - acc: 0.9184 - val_loss: 0.2553 - val_acc: 0.9064
# Epoch 8/50
# 60000/60000 [==============================] - 26s 440us/sample - loss: 0.2095 - acc: 0.9229 - val_loss: 0.2541 - val_acc: 0.9087
# Epoch 9/50
# 60000/60000 [==============================] - 26s 441us/sample - loss: 0.2002 - acc: 0.9263 - val_loss: 0.2469 - val_acc: 0.9090
# Epoch 10/50
# 60000/60000 [==============================] - 26s 440us/sample - loss: 0.1912 - acc: 0.9302 - val_loss: 0.2459 - val_acc: 0.9127
# Epoch 11/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.1845 - acc: 0.9324 - val_loss: 0.2398 - val_acc: 0.9139
# Epoch 12/50
# 60000/60000 [==============================] - 26s 438us/sample - loss: 0.1760 - acc: 0.9362 - val_loss: 0.2381 - val_acc: 0.9157
# Epoch 13/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.1700 - acc: 0.9377 - val_loss: 0.2364 - val_acc: 0.9157
# Epoch 14/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.1619 - acc: 0.9407 - val_loss: 0.2347 - val_acc: 0.9164
# Epoch 15/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.1578 - acc: 0.9423 - val_loss: 0.2338 - val_acc: 0.9183
# Epoch 16/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.1489 - acc: 0.9454 - val_loss: 0.2301 - val_acc: 0.9197
# Epoch 17/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.1462 - acc: 0.9474 - val_loss: 0.2277 - val_acc: 0.9206
# Epoch 18/50
# 60000/60000 [==============================] - 26s 441us/sample - loss: 0.1404 - acc: 0.9489 - val_loss: 0.2321 - val_acc: 0.9179
# Epoch 19/50
# 60000/60000 [==============================] - 26s 441us/sample - loss: 0.1353 - acc: 0.9498 - val_loss: 0.2324 - val_acc: 0.9191
# Epoch 20/50
# 60000/60000 [==============================] - 26s 442us/sample - loss: 0.1302 - acc: 0.9533 - val_loss: 0.2326 - val_acc: 0.9203
# Epoch 21/50
# 60000/60000 [==============================] - 26s 441us/sample - loss: 0.1258 - acc: 0.9541 - val_loss: 0.2268 - val_acc: 0.9210
# Epoch 22/50
# 60000/60000 [==============================] - 26s 440us/sample - loss: 0.1215 - acc: 0.9555 - val_loss: 0.2319 - val_acc: 0.9210
# Epoch 23/50
# 60000/60000 [==============================] - 27s 444us/sample - loss: 0.1183 - acc: 0.9567 - val_loss: 0.2348 - val_acc: 0.9223
# Epoch 24/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.1144 - acc: 0.9588 - val_loss: 0.2293 - val_acc: 0.9233
# Epoch 25/50
# 60000/60000 [==============================] - 27s 444us/sample - loss: 0.1103 - acc: 0.9606 - val_loss: 0.2266 - val_acc: 0.9243
# Epoch 26/50
# 60000/60000 [==============================] - 26s 440us/sample - loss: 0.1070 - acc: 0.9615 - val_loss: 0.2300 - val_acc: 0.9224
# Epoch 27/50
# 60000/60000 [==============================] - 26s 441us/sample - loss: 0.1040 - acc: 0.9631 - val_loss: 0.2307 - val_acc: 0.9229
# Epoch 28/50
# 60000/60000 [==============================] - 26s 440us/sample - loss: 0.0991 - acc: 0.9651 - val_loss: 0.2319 - val_acc: 0.9219
# Epoch 29/50
# 60000/60000 [==============================] - 26s 440us/sample - loss: 0.0970 - acc: 0.9658 - val_loss: 0.2296 - val_acc: 0.9238
# Epoch 30/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.0940 - acc: 0.9662 - val_loss: 0.2275 - val_acc: 0.9251
# Epoch 31/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.0907 - acc: 0.9679 - val_loss: 0.2292 - val_acc: 0.9252
# Epoch 32/50
# 60000/60000 [==============================] - 26s 440us/sample - loss: 0.0882 - acc: 0.9682 - val_loss: 0.2369 - val_acc: 0.9224
# Epoch 33/50
# 60000/60000 [==============================] - 26s 440us/sample - loss: 0.0855 - acc: 0.9696 - val_loss: 0.2366 - val_acc: 0.9243
# Epoch 34/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.0821 - acc: 0.9705 - val_loss: 0.2415 - val_acc: 0.9248
# Epoch 35/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.0808 - acc: 0.9710 - val_loss: 0.2374 - val_acc: 0.9238
# Epoch 36/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.0792 - acc: 0.9713 - val_loss: 0.2369 - val_acc: 0.9252
# Epoch 37/50
# 60000/60000 [==============================] - 27s 443us/sample - loss: 0.0765 - acc: 0.9728 - val_loss: 0.2386 - val_acc: 0.9260
# Epoch 38/50
# 60000/60000 [==============================] - 27s 445us/sample - loss: 0.0745 - acc: 0.9739 - val_loss: 0.2393 - val_acc: 0.9259
# Epoch 39/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.0714 - acc: 0.9748 - val_loss: 0.2416 - val_acc: 0.9266
# Epoch 40/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.0697 - acc: 0.9753 - val_loss: 0.2445 - val_acc: 0.9242
# Epoch 41/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.0689 - acc: 0.9752 - val_loss: 0.2441 - val_acc: 0.9259
# Epoch 42/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.0674 - acc: 0.9763 - val_loss: 0.2459 - val_acc: 0.9247
# Epoch 43/50
# 60000/60000 [==============================] - 26s 440us/sample - loss: 0.0644 - acc: 0.9776 - val_loss: 0.2407 - val_acc: 0.9266
# Epoch 44/50
# 60000/60000 [==============================] - 26s 440us/sample - loss: 0.0628 - acc: 0.9778 - val_loss: 0.2439 - val_acc: 0.9263
# Epoch 45/50
# 60000/60000 [==============================] - 26s 441us/sample - loss: 0.0607 - acc: 0.9782 - val_loss: 0.2426 - val_acc: 0.9257
# Epoch 46/50
# 60000/60000 [==============================] - 26s 440us/sample - loss: 0.0595 - acc: 0.9796 - val_loss: 0.2480 - val_acc: 0.9251
# Epoch 47/50
# 60000/60000 [==============================] - 26s 440us/sample - loss: 0.0585 - acc: 0.9794 - val_loss: 0.2513 - val_acc: 0.9285
# Epoch 48/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.0558 - acc: 0.9804 - val_loss: 0.2479 - val_acc: 0.9291
# Epoch 49/50
# 60000/60000 [==============================] - 26s 440us/sample - loss: 0.0546 - acc: 0.9812 - val_loss: 0.2470 - val_acc: 0.9249
# Epoch 50/50
# 60000/60000 [==============================] - 26s 439us/sample - loss: 0.0554 - acc: 0.9802 - val_loss: 0.2557 - val_acc: 0.9255
# 10000/10000 [==============================] - 1s 135us/sample - loss: 0.2557 - acc: 0.9255






