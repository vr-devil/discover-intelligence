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
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPool2D((2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(1024, activation='relu'),
    layers.Dense(1024, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

# Train on 60000 samples, validate on 10000 samples
# Epoch 1/100
# 60000/60000 [==============================] - 30s 497us/sample - loss: 0.4176 - acc: 0.8524 - val_loss: 0.3011 - val_acc: 0.8926
# Epoch 2/100
# 60000/60000 [==============================] - 28s 462us/sample - loss: 0.2816 - acc: 0.8979 - val_loss: 0.2667 - val_acc: 0.9009
# Epoch 3/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.2478 - acc: 0.9090 - val_loss: 0.2569 - val_acc: 0.9061
# Epoch 4/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.2270 - acc: 0.9176 - val_loss: 0.2525 - val_acc: 0.9072
# Epoch 5/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.2078 - acc: 0.9244 - val_loss: 0.2400 - val_acc: 0.9117
# Epoch 6/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.1957 - acc: 0.9280 - val_loss: 0.2327 - val_acc: 0.9160
# Epoch 7/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.1819 - acc: 0.9339 - val_loss: 0.2318 - val_acc: 0.9164
# Epoch 8/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.1736 - acc: 0.9367 - val_loss: 0.2281 - val_acc: 0.9205
# Epoch 9/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.1627 - acc: 0.9412 - val_loss: 0.2301 - val_acc: 0.9195
# Epoch 10/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.1549 - acc: 0.9433 - val_loss: 0.2321 - val_acc: 0.9195
# Epoch 11/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.1475 - acc: 0.9470 - val_loss: 0.2267 - val_acc: 0.9220
# Epoch 12/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.1413 - acc: 0.9492 - val_loss: 0.2337 - val_acc: 0.9190
# Epoch 13/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.1348 - acc: 0.9510 - val_loss: 0.2267 - val_acc: 0.9220
# Epoch 14/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.1265 - acc: 0.9540 - val_loss: 0.2281 - val_acc: 0.9223
# Epoch 15/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.1218 - acc: 0.9565 - val_loss: 0.2281 - val_acc: 0.9227
# Epoch 16/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.1161 - acc: 0.9581 - val_loss: 0.2280 - val_acc: 0.9234
# Epoch 17/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.1117 - acc: 0.9596 - val_loss: 0.2287 - val_acc: 0.9249
# Epoch 18/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.1064 - acc: 0.9615 - val_loss: 0.2283 - val_acc: 0.9246
# Epoch 19/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.1020 - acc: 0.9629 - val_loss: 0.2305 - val_acc: 0.9254
# Epoch 20/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0991 - acc: 0.9642 - val_loss: 0.2327 - val_acc: 0.9233
# Epoch 21/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0945 - acc: 0.9661 - val_loss: 0.2275 - val_acc: 0.9249
# Epoch 22/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0893 - acc: 0.9683 - val_loss: 0.2274 - val_acc: 0.9259
# Epoch 23/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0875 - acc: 0.9684 - val_loss: 0.2366 - val_acc: 0.9243
# Epoch 24/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0838 - acc: 0.9702 - val_loss: 0.2383 - val_acc: 0.9251
# Epoch 25/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0800 - acc: 0.9709 - val_loss: 0.2298 - val_acc: 0.9265
# Epoch 26/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0763 - acc: 0.9727 - val_loss: 0.2427 - val_acc: 0.9228
# Epoch 27/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0729 - acc: 0.9746 - val_loss: 0.2397 - val_acc: 0.9266
# Epoch 28/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0713 - acc: 0.9746 - val_loss: 0.2370 - val_acc: 0.9278
# Epoch 29/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0668 - acc: 0.9769 - val_loss: 0.2413 - val_acc: 0.9284
# Epoch 30/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0641 - acc: 0.9774 - val_loss: 0.2429 - val_acc: 0.9284
# Epoch 31/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0625 - acc: 0.9782 - val_loss: 0.2427 - val_acc: 0.9273
# Epoch 32/100
# 60000/60000 [==============================] - 28s 467us/sample - loss: 0.0612 - acc: 0.9785 - val_loss: 0.2433 - val_acc: 0.9267
# Epoch 33/100
# 60000/60000 [==============================] - 28s 466us/sample - loss: 0.0590 - acc: 0.9798 - val_loss: 0.2525 - val_acc: 0.9264
# Epoch 34/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0560 - acc: 0.9809 - val_loss: 0.2482 - val_acc: 0.9281
# Epoch 35/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0547 - acc: 0.9809 - val_loss: 0.2557 - val_acc: 0.9246
# Epoch 36/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0517 - acc: 0.9822 - val_loss: 0.2557 - val_acc: 0.9254
# Epoch 37/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0495 - acc: 0.9826 - val_loss: 0.2523 - val_acc: 0.9298
# Epoch 38/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0492 - acc: 0.9831 - val_loss: 0.2565 - val_acc: 0.9269
# Epoch 39/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0474 - acc: 0.9840 - val_loss: 0.2580 - val_acc: 0.9277
# Epoch 40/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0458 - acc: 0.9847 - val_loss: 0.2591 - val_acc: 0.9271
# Epoch 41/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0431 - acc: 0.9855 - val_loss: 0.2636 - val_acc: 0.9266
# Epoch 42/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0425 - acc: 0.9855 - val_loss: 0.2602 - val_acc: 0.9284
# Epoch 43/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0411 - acc: 0.9859 - val_loss: 0.2671 - val_acc: 0.9275
# Epoch 44/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0399 - acc: 0.9863 - val_loss: 0.2639 - val_acc: 0.9276
# Epoch 45/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0393 - acc: 0.9868 - val_loss: 0.2636 - val_acc: 0.9273
# Epoch 46/100
# 60000/60000 [==============================] - 28s 466us/sample - loss: 0.0376 - acc: 0.9873 - val_loss: 0.2683 - val_acc: 0.9291
# Epoch 47/100
# 60000/60000 [==============================] - 28s 465us/sample - loss: 0.0374 - acc: 0.9872 - val_loss: 0.2691 - val_acc: 0.9266
# Epoch 48/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0348 - acc: 0.9884 - val_loss: 0.2733 - val_acc: 0.9277
# Epoch 49/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0349 - acc: 0.9884 - val_loss: 0.2700 - val_acc: 0.9280
# Epoch 50/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0333 - acc: 0.9891 - val_loss: 0.2807 - val_acc: 0.9270
# Epoch 51/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0315 - acc: 0.9896 - val_loss: 0.2746 - val_acc: 0.9278
# Epoch 52/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0309 - acc: 0.9899 - val_loss: 0.2804 - val_acc: 0.9281
# Epoch 53/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0298 - acc: 0.9898 - val_loss: 0.2818 - val_acc: 0.9275
# Epoch 54/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0305 - acc: 0.9895 - val_loss: 0.2793 - val_acc: 0.9280
# Epoch 55/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0293 - acc: 0.9909 - val_loss: 0.2827 - val_acc: 0.9295
# Epoch 56/100
# 60000/60000 [==============================] - 28s 466us/sample - loss: 0.0290 - acc: 0.9905 - val_loss: 0.2914 - val_acc: 0.9283
# Epoch 57/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0285 - acc: 0.9908 - val_loss: 0.2821 - val_acc: 0.9284
# Epoch 58/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0272 - acc: 0.9911 - val_loss: 0.2820 - val_acc: 0.9287
# Epoch 59/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0259 - acc: 0.9916 - val_loss: 0.2887 - val_acc: 0.9277
# Epoch 60/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0256 - acc: 0.9921 - val_loss: 0.2926 - val_acc: 0.9276
# Epoch 61/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0246 - acc: 0.9925 - val_loss: 0.2868 - val_acc: 0.9285
# Epoch 62/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0239 - acc: 0.9926 - val_loss: 0.2930 - val_acc: 0.9278
# Epoch 63/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0248 - acc: 0.9919 - val_loss: 0.2914 - val_acc: 0.9278
# Epoch 64/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0226 - acc: 0.9930 - val_loss: 0.2919 - val_acc: 0.9287
# Epoch 65/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0224 - acc: 0.9931 - val_loss: 0.2929 - val_acc: 0.9282
# Epoch 66/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0229 - acc: 0.9925 - val_loss: 0.2953 - val_acc: 0.9288
# Epoch 67/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0212 - acc: 0.9931 - val_loss: 0.2947 - val_acc: 0.9272
# Epoch 68/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0215 - acc: 0.9934 - val_loss: 0.2957 - val_acc: 0.9272
# Epoch 69/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0214 - acc: 0.9932 - val_loss: 0.2929 - val_acc: 0.9287
# Epoch 70/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0209 - acc: 0.9937 - val_loss: 0.2954 - val_acc: 0.9288
# Epoch 71/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0197 - acc: 0.9937 - val_loss: 0.2982 - val_acc: 0.9285
# Epoch 72/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0199 - acc: 0.9934 - val_loss: 0.3035 - val_acc: 0.9288
# Epoch 73/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0189 - acc: 0.9942 - val_loss: 0.3073 - val_acc: 0.9292
# Epoch 74/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0188 - acc: 0.9944 - val_loss: 0.3067 - val_acc: 0.9285
# Epoch 75/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0179 - acc: 0.9947 - val_loss: 0.3045 - val_acc: 0.9289
# Epoch 76/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0185 - acc: 0.9944 - val_loss: 0.3205 - val_acc: 0.9283
# Epoch 77/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0174 - acc: 0.9948 - val_loss: 0.3157 - val_acc: 0.9290
# Epoch 78/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0170 - acc: 0.9949 - val_loss: 0.3116 - val_acc: 0.9287
# Epoch 79/100
# 60000/60000 [==============================] - 28s 465us/sample - loss: 0.0169 - acc: 0.9946 - val_loss: 0.3093 - val_acc: 0.9298
# Epoch 80/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0177 - acc: 0.9945 - val_loss: 0.3063 - val_acc: 0.9283
# Epoch 81/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0176 - acc: 0.9943 - val_loss: 0.3128 - val_acc: 0.9303
# Epoch 82/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0157 - acc: 0.9955 - val_loss: 0.3133 - val_acc: 0.9301
# Epoch 83/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0162 - acc: 0.9950 - val_loss: 0.3156 - val_acc: 0.9293
# Epoch 84/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0156 - acc: 0.9952 - val_loss: 0.3195 - val_acc: 0.9299
# Epoch 85/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0152 - acc: 0.9954 - val_loss: 0.3185 - val_acc: 0.9296
# Epoch 86/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0152 - acc: 0.9955 - val_loss: 0.3168 - val_acc: 0.9298
# Epoch 87/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0149 - acc: 0.9954 - val_loss: 0.3176 - val_acc: 0.9294
# Epoch 88/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0153 - acc: 0.9957 - val_loss: 0.3163 - val_acc: 0.9312
# Epoch 89/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0145 - acc: 0.9956 - val_loss: 0.3254 - val_acc: 0.9288
# Epoch 90/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0140 - acc: 0.9957 - val_loss: 0.3247 - val_acc: 0.9291
# Epoch 91/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0136 - acc: 0.9962 - val_loss: 0.3211 - val_acc: 0.9283
# Epoch 92/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0138 - acc: 0.9958 - val_loss: 0.3217 - val_acc: 0.9286
# Epoch 93/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0138 - acc: 0.9959 - val_loss: 0.3318 - val_acc: 0.9285
# Epoch 94/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0128 - acc: 0.9963 - val_loss: 0.3231 - val_acc: 0.9292
# Epoch 95/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0138 - acc: 0.9958 - val_loss: 0.3309 - val_acc: 0.9308
# Epoch 96/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0130 - acc: 0.9962 - val_loss: 0.3260 - val_acc: 0.9301
# Epoch 97/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0122 - acc: 0.9962 - val_loss: 0.3283 - val_acc: 0.9295
# Epoch 98/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0124 - acc: 0.9963 - val_loss: 0.3298 - val_acc: 0.9311
# Epoch 99/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0121 - acc: 0.9963 - val_loss: 0.3283 - val_acc: 0.9296
# Epoch 100/100
# 60000/60000 [==============================] - 28s 464us/sample - loss: 0.0125 - acc: 0.9962 - val_loss: 0.3372 - val_acc: 0.9289
# 10000/10000 [==============================] - 1s 145us/sample - loss: 0.3372 - acc: 0.9289







