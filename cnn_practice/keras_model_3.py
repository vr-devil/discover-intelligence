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
    layers.Conv2D(96, (2, 2), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPool2D((2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(256, (2, 2), activation='relu'),
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
# 60000/60000 [==============================] - 50s 830us/sample - loss: 0.3674 - acc: 0.8719 - val_loss: 0.2534 - val_acc: 0.9074
# Epoch 2/50
# 60000/60000 [==============================] - 47s 777us/sample - loss: 0.2032 - acc: 0.9258 - val_loss: 0.2524 - val_acc: 0.9097
# Epoch 3/50
# 60000/60000 [==============================] - 48s 797us/sample - loss: 0.1448 - acc: 0.9475 - val_loss: 0.2368 - val_acc: 0.9200
# Epoch 4/50
# 60000/60000 [==============================] - 47s 778us/sample - loss: 0.1030 - acc: 0.9634 - val_loss: 0.2504 - val_acc: 0.9153
# Epoch 5/50
# 60000/60000 [==============================] - 47s 779us/sample - loss: 0.0731 - acc: 0.9750 - val_loss: 0.2543 - val_acc: 0.9203
# Epoch 6/50
# 60000/60000 [==============================] - 47s 781us/sample - loss: 0.0507 - acc: 0.9837 - val_loss: 0.2614 - val_acc: 0.9208
# Epoch 7/50
# 60000/60000 [==============================] - 47s 777us/sample - loss: 0.0377 - acc: 0.9885 - val_loss: 0.2724 - val_acc: 0.9195
# Epoch 8/50
# 60000/60000 [==============================] - 48s 794us/sample - loss: 0.0283 - acc: 0.9923 - val_loss: 0.2811 - val_acc: 0.9202
# Epoch 9/50
# 60000/60000 [==============================] - 47s 785us/sample - loss: 0.0215 - acc: 0.9941 - val_loss: 0.2830 - val_acc: 0.9220
# Epoch 10/50
# 60000/60000 [==============================] - 47s 781us/sample - loss: 0.0160 - acc: 0.9965 - val_loss: 0.2923 - val_acc: 0.9230
# Epoch 11/50
# 60000/60000 [==============================] - 47s 781us/sample - loss: 0.0139 - acc: 0.9966 - val_loss: 0.3011 - val_acc: 0.9224
# Epoch 12/50
# 60000/60000 [==============================] - 47s 777us/sample - loss: 0.0104 - acc: 0.9979 - val_loss: 0.3017 - val_acc: 0.9246
# Epoch 13/50
# 60000/60000 [==============================] - 47s 790us/sample - loss: 0.0093 - acc: 0.9981 - val_loss: 0.3053 - val_acc: 0.9219
# Epoch 14/50
# 60000/60000 [==============================] - 48s 798us/sample - loss: 0.0079 - acc: 0.9985 - val_loss: 0.3394 - val_acc: 0.9171
# Epoch 15/50
# 60000/60000 [==============================] - 47s 787us/sample - loss: 0.0068 - acc: 0.9988 - val_loss: 0.3145 - val_acc: 0.9241
# Epoch 16/50
# 60000/60000 [==============================] - 47s 787us/sample - loss: 0.0057 - acc: 0.9991 - val_loss: 0.3169 - val_acc: 0.9248
# Epoch 17/50
# 60000/60000 [==============================] - 47s 778us/sample - loss: 0.0051 - acc: 0.9993 - val_loss: 0.3248 - val_acc: 0.9242
# Epoch 18/50
# 60000/60000 [==============================] - 47s 788us/sample - loss: 0.0048 - acc: 0.9993 - val_loss: 0.3243 - val_acc: 0.9223
# Epoch 19/50
# 60000/60000 [==============================] - 47s 780us/sample - loss: 0.0037 - acc: 0.9995 - val_loss: 0.3278 - val_acc: 0.9227
# Epoch 20/50
# 60000/60000 [==============================] - 47s 781us/sample - loss: 0.0037 - acc: 0.9995 - val_loss: 0.3258 - val_acc: 0.9254
# Epoch 21/50
# 60000/60000 [==============================] - 47s 782us/sample - loss: 0.0033 - acc: 0.9995 - val_loss: 0.3327 - val_acc: 0.9239
# Epoch 22/50
# 60000/60000 [==============================] - 47s 780us/sample - loss: 0.0037 - acc: 0.9993 - val_loss: 0.3323 - val_acc: 0.9249
# Epoch 23/50
# 60000/60000 [==============================] - 46s 770us/sample - loss: 0.0033 - acc: 0.9996 - val_loss: 0.3373 - val_acc: 0.9245
# Epoch 24/50
# 60000/60000 [==============================] - 46s 772us/sample - loss: 0.0028 - acc: 0.9997 - val_loss: 0.3361 - val_acc: 0.9241
# Epoch 25/50
# 60000/60000 [==============================] - 46s 774us/sample - loss: 0.0028 - acc: 0.9997 - val_loss: 0.3363 - val_acc: 0.9246
# Epoch 26/50
# 60000/60000 [==============================] - 47s 785us/sample - loss: 0.0021 - acc: 0.9998 - val_loss: 0.3375 - val_acc: 0.9249
# Epoch 27/50
# 60000/60000 [==============================] - 46s 771us/sample - loss: 0.0024 - acc: 0.9997 - val_loss: 0.3406 - val_acc: 0.9236
# Epoch 28/50
# 60000/60000 [==============================] - 47s 784us/sample - loss: 0.0022 - acc: 0.9997 - val_loss: 0.3419 - val_acc: 0.9239
# Epoch 29/50
# 60000/60000 [==============================] - 47s 785us/sample - loss: 0.0017 - acc: 0.9999 - val_loss: 0.3504 - val_acc: 0.9243
# Epoch 30/50
# 60000/60000 [==============================] - 47s 777us/sample - loss: 0.0019 - acc: 0.9998 - val_loss: 0.3488 - val_acc: 0.9253
# Epoch 31/50
# 60000/60000 [==============================] - 47s 784us/sample - loss: 0.0016 - acc: 0.9999 - val_loss: 0.3472 - val_acc: 0.9257
# Epoch 32/50
# 60000/60000 [==============================] - 48s 802us/sample - loss: 0.0016 - acc: 0.9999 - val_loss: 0.3485 - val_acc: 0.9253
# Epoch 33/50
# 60000/60000 [==============================] - 48s 801us/sample - loss: 0.0016 - acc: 0.9998 - val_loss: 0.3517 - val_acc: 0.9250
# Epoch 34/50
# 60000/60000 [==============================] - 47s 780us/sample - loss: 0.0014 - acc: 0.9999 - val_loss: 0.3563 - val_acc: 0.9249
# Epoch 35/50
# 60000/60000 [==============================] - 47s 786us/sample - loss: 0.0015 - acc: 0.9999 - val_loss: 0.3581 - val_acc: 0.9262
# Epoch 36/50
# 60000/60000 [==============================] - 48s 795us/sample - loss: 0.0015 - acc: 0.9998 - val_loss: 0.3631 - val_acc: 0.9251
# Epoch 37/50
# 60000/60000 [==============================] - 46s 772us/sample - loss: 0.0015 - acc: 0.9998 - val_loss: 0.3633 - val_acc: 0.9243
# Epoch 38/50
# 60000/60000 [==============================] - 46s 773us/sample - loss: 0.0012 - acc: 0.9999 - val_loss: 0.3577 - val_acc: 0.9238
# Epoch 39/50
# 60000/60000 [==============================] - 46s 771us/sample - loss: 0.0012 - acc: 0.9999 - val_loss: 0.3549 - val_acc: 0.9257
# Epoch 40/50
# 60000/60000 [==============================] - 46s 770us/sample - loss: 9.5390e-04 - acc: 1.0000 - val_loss: 0.3585 - val_acc: 0.9263
# Epoch 41/50
# 60000/60000 [==============================] - 46s 771us/sample - loss: 0.0010 - acc: 0.9999 - val_loss: 0.3579 - val_acc: 0.9255
# Epoch 42/50
# 60000/60000 [==============================] - 46s 772us/sample - loss: 9.4747e-04 - acc: 1.0000 - val_loss: 0.3600 - val_acc: 0.9263
# Epoch 43/50
# 60000/60000 [==============================] - 46s 772us/sample - loss: 9.4614e-04 - acc: 0.9999 - val_loss: 0.3615 - val_acc: 0.9243
# Epoch 44/50
# 60000/60000 [==============================] - 46s 771us/sample - loss: 9.2876e-04 - acc: 1.0000 - val_loss: 0.3613 - val_acc: 0.9241
# Epoch 45/50
# 60000/60000 [==============================] - 46s 771us/sample - loss: 8.2986e-04 - acc: 0.9999 - val_loss: 0.3623 - val_acc: 0.9258
# Epoch 46/50
# 60000/60000 [==============================] - 46s 772us/sample - loss: 8.8443e-04 - acc: 0.9999 - val_loss: 0.3701 - val_acc: 0.9246
# Epoch 47/50
# 60000/60000 [==============================] - 46s 771us/sample - loss: 0.0013 - acc: 0.9998 - val_loss: 0.3698 - val_acc: 0.9251
# Epoch 48/50
# 60000/60000 [==============================] - 46s 771us/sample - loss: 0.0010 - acc: 0.9999 - val_loss: 0.3692 - val_acc: 0.9223
# Epoch 49/50
# 60000/60000 [==============================] - 46s 771us/sample - loss: 8.2002e-04 - acc: 1.0000 - val_loss: 0.3728 - val_acc: 0.9236
# Epoch 50/50
# 60000/60000 [==============================] - 46s 771us/sample - loss: 7.4811e-04 - acc: 1.0000 - val_loss: 0.3722 - val_acc: 0.9245
# 10000/10000 [==============================] - 1s 133us/sample - loss: 0.3722 - acc: 0.9245




