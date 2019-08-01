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
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.4),
    layers.Flatten(),
    layers.Dense(2048, activation='relu'),
    layers.Dense(2048, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

# Train on 60000 samples, validate on 10000 samples
# Epoch 1/50
# 60000/60000 [==============================] - 39s 644us/sample - loss: 0.5464 - acc: 0.8292 - val_loss: 0.3457 - val_acc: 0.8751
# Epoch 2/50
# 60000/60000 [==============================] - 38s 641us/sample - loss: 0.3218 - acc: 0.8815 - val_loss: 0.3008 - val_acc: 0.8918
# Epoch 3/50
# 60000/60000 [==============================] - 37s 620us/sample - loss: 0.2805 - acc: 0.8981 - val_loss: 0.2921 - val_acc: 0.8936
# Epoch 4/50
# 60000/60000 [==============================] - 37s 617us/sample - loss: 0.2531 - acc: 0.9065 - val_loss: 0.2814 - val_acc: 0.8960
# Epoch 5/50
# 60000/60000 [==============================] - 38s 629us/sample - loss: 0.2350 - acc: 0.9143 - val_loss: 0.2553 - val_acc: 0.9069
# Epoch 6/50
# 60000/60000 [==============================] - 36s 606us/sample - loss: 0.2184 - acc: 0.9204 - val_loss: 0.2531 - val_acc: 0.9084
# Epoch 7/50
# 60000/60000 [==============================] - 36s 604us/sample - loss: 0.2029 - acc: 0.9248 - val_loss: 0.2499 - val_acc: 0.9120
# Epoch 8/50
# 60000/60000 [==============================] - 37s 616us/sample - loss: 0.1919 - acc: 0.9300 - val_loss: 0.2392 - val_acc: 0.9134
# Epoch 9/50
# 60000/60000 [==============================] - 37s 616us/sample - loss: 0.1815 - acc: 0.9336 - val_loss: 0.2362 - val_acc: 0.9154
# Epoch 10/50
# 60000/60000 [==============================] - 36s 607us/sample - loss: 0.1713 - acc: 0.9374 - val_loss: 0.2348 - val_acc: 0.9172
# Epoch 11/50
# 60000/60000 [==============================] - 36s 604us/sample - loss: 0.1628 - acc: 0.9403 - val_loss: 0.2365 - val_acc: 0.9165
# Epoch 12/50
# 60000/60000 [==============================] - 36s 605us/sample - loss: 0.1528 - acc: 0.9444 - val_loss: 0.2404 - val_acc: 0.9147
# Epoch 13/50
# 60000/60000 [==============================] - 36s 605us/sample - loss: 0.1427 - acc: 0.9476 - val_loss: 0.2338 - val_acc: 0.9205
# Epoch 14/50
# 60000/60000 [==============================] - 38s 633us/sample - loss: 0.1385 - acc: 0.9487 - val_loss: 0.2311 - val_acc: 0.9204
# Epoch 15/50
# 60000/60000 [==============================] - 36s 608us/sample - loss: 0.1300 - acc: 0.9523 - val_loss: 0.2302 - val_acc: 0.9208
# Epoch 16/50
# 60000/60000 [==============================] - 36s 604us/sample - loss: 0.1234 - acc: 0.9553 - val_loss: 0.2317 - val_acc: 0.9213
# Epoch 17/50
# 60000/60000 [==============================] - 36s 605us/sample - loss: 0.1162 - acc: 0.9574 - val_loss: 0.2403 - val_acc: 0.9216
# Epoch 18/50
# 60000/60000 [==============================] - 36s 607us/sample - loss: 0.1111 - acc: 0.9593 - val_loss: 0.2357 - val_acc: 0.9217
# Epoch 19/50
# 60000/60000 [==============================] - 36s 605us/sample - loss: 0.1056 - acc: 0.9616 - val_loss: 0.2322 - val_acc: 0.9239
# Epoch 20/50
# 60000/60000 [==============================] - 36s 605us/sample - loss: 0.1005 - acc: 0.9634 - val_loss: 0.2393 - val_acc: 0.9221
# Epoch 21/50
# 60000/60000 [==============================] - 36s 608us/sample - loss: 0.0948 - acc: 0.9658 - val_loss: 0.2407 - val_acc: 0.9241
# Epoch 22/50
# 60000/60000 [==============================] - 36s 608us/sample - loss: 0.0909 - acc: 0.9672 - val_loss: 0.2424 - val_acc: 0.9239
# Epoch 23/50
# 60000/60000 [==============================] - 37s 609us/sample - loss: 0.0869 - acc: 0.9684 - val_loss: 0.2430 - val_acc: 0.9222
# Epoch 24/50
# 60000/60000 [==============================] - 37s 609us/sample - loss: 0.0812 - acc: 0.9707 - val_loss: 0.2515 - val_acc: 0.9245
# Epoch 25/50
# 60000/60000 [==============================] - 36s 608us/sample - loss: 0.0776 - acc: 0.9725 - val_loss: 0.2413 - val_acc: 0.9258
# Epoch 26/50
# 60000/60000 [==============================] - 36s 606us/sample - loss: 0.0721 - acc: 0.9747 - val_loss: 0.2485 - val_acc: 0.9243
# Epoch 27/50
# 60000/60000 [==============================] - 36s 605us/sample - loss: 0.0700 - acc: 0.9753 - val_loss: 0.2618 - val_acc: 0.9238
# Epoch 28/50
# 60000/60000 [==============================] - 36s 606us/sample - loss: 0.0661 - acc: 0.9758 - val_loss: 0.2545 - val_acc: 0.9248
# Epoch 29/50
# 60000/60000 [==============================] - 36s 606us/sample - loss: 0.0629 - acc: 0.9774 - val_loss: 0.2527 - val_acc: 0.9256
# Epoch 30/50
# 60000/60000 [==============================] - 36s 606us/sample - loss: 0.0603 - acc: 0.9791 - val_loss: 0.2543 - val_acc: 0.9251
# Epoch 31/50
# 60000/60000 [==============================] - 36s 608us/sample - loss: 0.0571 - acc: 0.9801 - val_loss: 0.2642 - val_acc: 0.9242
# Epoch 32/50
# 60000/60000 [==============================] - 37s 610us/sample - loss: 0.0547 - acc: 0.9808 - val_loss: 0.2689 - val_acc: 0.9264
# Epoch 33/50
# 60000/60000 [==============================] - 37s 611us/sample - loss: 0.0532 - acc: 0.9806 - val_loss: 0.2680 - val_acc: 0.9249
# Epoch 34/50
# 60000/60000 [==============================] - 37s 614us/sample - loss: 0.0495 - acc: 0.9826 - val_loss: 0.2665 - val_acc: 0.9274
# Epoch 35/50
# 60000/60000 [==============================] - 37s 621us/sample - loss: 0.0477 - acc: 0.9833 - val_loss: 0.2714 - val_acc: 0.9272
# Epoch 36/50
# 60000/60000 [==============================] - 37s 611us/sample - loss: 0.0451 - acc: 0.9845 - val_loss: 0.2827 - val_acc: 0.9266
# Epoch 37/50
# 60000/60000 [==============================] - 37s 610us/sample - loss: 0.0432 - acc: 0.9851 - val_loss: 0.2726 - val_acc: 0.9257
# Epoch 38/50
# 60000/60000 [==============================] - 36s 608us/sample - loss: 0.0407 - acc: 0.9861 - val_loss: 0.2804 - val_acc: 0.9261
# Epoch 39/50
# 60000/60000 [==============================] - 36s 608us/sample - loss: 0.0381 - acc: 0.9873 - val_loss: 0.2756 - val_acc: 0.9291
# Epoch 40/50
# 60000/60000 [==============================] - 36s 607us/sample - loss: 0.0379 - acc: 0.9866 - val_loss: 0.2866 - val_acc: 0.9258
# Epoch 41/50
# 60000/60000 [==============================] - 37s 612us/sample - loss: 0.0342 - acc: 0.9886 - val_loss: 0.2903 - val_acc: 0.9274
# Epoch 42/50
# 60000/60000 [==============================] - 36s 608us/sample - loss: 0.0353 - acc: 0.9882 - val_loss: 0.2822 - val_acc: 0.9264
# Epoch 43/50
# 60000/60000 [==============================] - 36s 608us/sample - loss: 0.0323 - acc: 0.9896 - val_loss: 0.2967 - val_acc: 0.9249
# Epoch 44/50
# 60000/60000 [==============================] - 36s 607us/sample - loss: 0.0323 - acc: 0.9890 - val_loss: 0.2942 - val_acc: 0.9278
# Epoch 45/50
# 60000/60000 [==============================] - 36s 608us/sample - loss: 0.0317 - acc: 0.9893 - val_loss: 0.2971 - val_acc: 0.9277
# Epoch 46/50
# 60000/60000 [==============================] - 36s 607us/sample - loss: 0.0300 - acc: 0.9902 - val_loss: 0.2875 - val_acc: 0.9273
# Epoch 47/50
# 60000/60000 [==============================] - 36s 607us/sample - loss: 0.0289 - acc: 0.9906 - val_loss: 0.2946 - val_acc: 0.9282
# Epoch 48/50
# 60000/60000 [==============================] - 36s 607us/sample - loss: 0.0266 - acc: 0.9913 - val_loss: 0.3022 - val_acc: 0.9265
# Epoch 49/50
# 60000/60000 [==============================] - 36s 607us/sample - loss: 0.0267 - acc: 0.9909 - val_loss: 0.3180 - val_acc: 0.9260
# Epoch 50/50
# 60000/60000 [==============================] - 36s 607us/sample - loss: 0.0262 - acc: 0.9913 - val_loss: 0.3090 - val_acc: 0.9271
# 10000/10000 [==============================] - 2s 172us/sample - loss: 0.3090 - acc: 0.9271





