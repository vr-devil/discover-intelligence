# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = np.reshape(x_train, (60000, 28, 28, 1))
x_test = np.reshape(x_test, (10000, 28, 28, 1))

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.models.Sequential([
    layers.Conv2D(96, (5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.MaxPool2D((2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(192, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

optimizer = tf.keras.optimizers.Adagrad(0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
writer.flush()

model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

writer.flush()

# Train on 60000 samples, validate on 10000 samples
# Epoch 1/100
# 60000/60000 [==============================] - 29s 491us/sample - loss: 0.4238 - acc: 0.8448 - val_loss: 0.2899 - val_acc: 0.8905
# Epoch 2/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.2947 - acc: 0.8919 - val_loss: 0.2566 - val_acc: 0.9060
# Epoch 3/100
# 60000/60000 [==============================] - 27s 454us/sample - loss: 0.2551 - acc: 0.9050 - val_loss: 0.2409 - val_acc: 0.9124
# Epoch 4/100
# 60000/60000 [==============================] - 27s 453us/sample - loss: 0.2284 - acc: 0.9150 - val_loss: 0.2371 - val_acc: 0.9145
# Epoch 5/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.2115 - acc: 0.9209 - val_loss: 0.2247 - val_acc: 0.9170
# Epoch 6/100
# 60000/60000 [==============================] - 28s 458us/sample - loss: 0.1956 - acc: 0.9262 - val_loss: 0.2298 - val_acc: 0.9162
# Epoch 7/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.1816 - acc: 0.9319 - val_loss: 0.1990 - val_acc: 0.9274
# Epoch 8/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.1729 - acc: 0.9353 - val_loss: 0.1991 - val_acc: 0.9274
# Epoch 9/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.1637 - acc: 0.9392 - val_loss: 0.2011 - val_acc: 0.9272
# Epoch 10/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.1519 - acc: 0.9444 - val_loss: 0.1986 - val_acc: 0.9299
# Epoch 11/100
# 60000/60000 [==============================] - 27s 454us/sample - loss: 0.1453 - acc: 0.9454 - val_loss: 0.1956 - val_acc: 0.9287
# Epoch 12/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.1362 - acc: 0.9488 - val_loss: 0.1919 - val_acc: 0.9338
# Epoch 13/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.1295 - acc: 0.9510 - val_loss: 0.1926 - val_acc: 0.9344
# Epoch 14/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.1200 - acc: 0.9550 - val_loss: 0.1890 - val_acc: 0.9336
# Epoch 15/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.1163 - acc: 0.9564 - val_loss: 0.1943 - val_acc: 0.9344
# Epoch 16/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.1102 - acc: 0.9584 - val_loss: 0.1952 - val_acc: 0.9334
# Epoch 17/100
# 60000/60000 [==============================] - 27s 454us/sample - loss: 0.1063 - acc: 0.9595 - val_loss: 0.2238 - val_acc: 0.9291
# Epoch 18/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0978 - acc: 0.9642 - val_loss: 0.1919 - val_acc: 0.9340
# Epoch 19/100
# 60000/60000 [==============================] - 27s 454us/sample - loss: 0.0928 - acc: 0.9649 - val_loss: 0.1962 - val_acc: 0.9357
# Epoch 20/100
# 60000/60000 [==============================] - 27s 453us/sample - loss: 0.0884 - acc: 0.9667 - val_loss: 0.2003 - val_acc: 0.9361
# Epoch 21/100
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.0845 - acc: 0.9688 - val_loss: 0.2023 - val_acc: 0.9350
# Epoch 22/100
# 60000/60000 [==============================] - 27s 454us/sample - loss: 0.0800 - acc: 0.9703 - val_loss: 0.2087 - val_acc: 0.9359
# Epoch 23/100
# 60000/60000 [==============================] - 27s 453us/sample - loss: 0.0771 - acc: 0.9715 - val_loss: 0.2018 - val_acc: 0.9363
# Epoch 24/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0724 - acc: 0.9726 - val_loss: 0.2065 - val_acc: 0.9381
# Epoch 25/100
# 60000/60000 [==============================] - 27s 454us/sample - loss: 0.0697 - acc: 0.9742 - val_loss: 0.2080 - val_acc: 0.9347
# Epoch 26/100
# 60000/60000 [==============================] - 27s 454us/sample - loss: 0.0653 - acc: 0.9755 - val_loss: 0.2153 - val_acc: 0.9352
# Epoch 27/100
# 60000/60000 [==============================] - 27s 453us/sample - loss: 0.0633 - acc: 0.9766 - val_loss: 0.2123 - val_acc: 0.9367
# Epoch 28/100
# 60000/60000 [==============================] - 28s 460us/sample - loss: 0.0605 - acc: 0.9773 - val_loss: 0.2273 - val_acc: 0.9354
# Epoch 29/100
# 60000/60000 [==============================] - 28s 468us/sample - loss: 0.0567 - acc: 0.9785 - val_loss: 0.2198 - val_acc: 0.9357
# Epoch 30/100
# 60000/60000 [==============================] - 27s 458us/sample - loss: 0.0515 - acc: 0.9810 - val_loss: 0.2152 - val_acc: 0.9382
# Epoch 31/100
# 60000/60000 [==============================] - 28s 459us/sample - loss: 0.0517 - acc: 0.9808 - val_loss: 0.2222 - val_acc: 0.9393
# Epoch 32/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0506 - acc: 0.9815 - val_loss: 0.2241 - val_acc: 0.9375
# Epoch 33/100
# 60000/60000 [==============================] - 27s 458us/sample - loss: 0.0461 - acc: 0.9831 - val_loss: 0.2200 - val_acc: 0.9378
# Epoch 34/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0460 - acc: 0.9829 - val_loss: 0.2214 - val_acc: 0.9372
# Epoch 35/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0416 - acc: 0.9851 - val_loss: 0.2266 - val_acc: 0.9390
# Epoch 36/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0397 - acc: 0.9856 - val_loss: 0.2377 - val_acc: 0.9363
# Epoch 37/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0397 - acc: 0.9855 - val_loss: 0.2307 - val_acc: 0.9391
# Epoch 38/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0386 - acc: 0.9865 - val_loss: 0.2364 - val_acc: 0.9382
# Epoch 39/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0369 - acc: 0.9866 - val_loss: 0.2327 - val_acc: 0.9387
# Epoch 40/100
# 60000/60000 [==============================] - 28s 459us/sample - loss: 0.0356 - acc: 0.9867 - val_loss: 0.2319 - val_acc: 0.9407
# Epoch 41/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0326 - acc: 0.9884 - val_loss: 0.2367 - val_acc: 0.9390
# Epoch 42/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0316 - acc: 0.9887 - val_loss: 0.2415 - val_acc: 0.9371
# Epoch 43/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0313 - acc: 0.9889 - val_loss: 0.2520 - val_acc: 0.9386
# Epoch 44/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0310 - acc: 0.9890 - val_loss: 0.2410 - val_acc: 0.9389
# Epoch 45/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.0296 - acc: 0.9897 - val_loss: 0.2430 - val_acc: 0.9416
# Epoch 46/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0279 - acc: 0.9899 - val_loss: 0.2474 - val_acc: 0.9425
# Epoch 47/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.0265 - acc: 0.9905 - val_loss: 0.2421 - val_acc: 0.9405
# Epoch 48/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0251 - acc: 0.9914 - val_loss: 0.2502 - val_acc: 0.9438
# Epoch 49/100
# 60000/60000 [==============================] - 28s 459us/sample - loss: 0.0247 - acc: 0.9914 - val_loss: 0.2517 - val_acc: 0.9404
# Epoch 50/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0232 - acc: 0.9920 - val_loss: 0.2493 - val_acc: 0.9415
# Epoch 51/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0211 - acc: 0.9928 - val_loss: 0.2518 - val_acc: 0.9409
# Epoch 52/100
# 60000/60000 [==============================] - 27s 458us/sample - loss: 0.0224 - acc: 0.9925 - val_loss: 0.2543 - val_acc: 0.9409
# Epoch 53/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0209 - acc: 0.9931 - val_loss: 0.2568 - val_acc: 0.9410
# Epoch 54/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0201 - acc: 0.9933 - val_loss: 0.2543 - val_acc: 0.9427
# Epoch 55/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0205 - acc: 0.9927 - val_loss: 0.2592 - val_acc: 0.9419
# Epoch 56/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.0200 - acc: 0.9929 - val_loss: 0.2637 - val_acc: 0.9393
# Epoch 57/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.0199 - acc: 0.9932 - val_loss: 0.2665 - val_acc: 0.9413
# Epoch 58/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0182 - acc: 0.9938 - val_loss: 0.2673 - val_acc: 0.9417
# Epoch 59/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0180 - acc: 0.9938 - val_loss: 0.2632 - val_acc: 0.9429
# Epoch 60/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.0178 - acc: 0.9941 - val_loss: 0.2709 - val_acc: 0.9410
# Epoch 61/100
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.0183 - acc: 0.9937 - val_loss: 0.2653 - val_acc: 0.9403
# Epoch 62/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0177 - acc: 0.9940 - val_loss: 0.2677 - val_acc: 0.9406
# Epoch 63/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0176 - acc: 0.9939 - val_loss: 0.2714 - val_acc: 0.9410
# Epoch 64/100
# 60000/60000 [==============================] - 27s 458us/sample - loss: 0.0174 - acc: 0.9941 - val_loss: 0.2773 - val_acc: 0.9383
# Epoch 65/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.0149 - acc: 0.9950 - val_loss: 0.2655 - val_acc: 0.9418
# Epoch 66/100
# 60000/60000 [==============================] - 27s 458us/sample - loss: 0.0157 - acc: 0.9946 - val_loss: 0.2683 - val_acc: 0.9409
# Epoch 67/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.0149 - acc: 0.9951 - val_loss: 0.2678 - val_acc: 0.9414
# Epoch 68/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0137 - acc: 0.9954 - val_loss: 0.2750 - val_acc: 0.9408
# Epoch 69/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.0144 - acc: 0.9950 - val_loss: 0.2719 - val_acc: 0.9418
# Epoch 70/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0135 - acc: 0.9958 - val_loss: 0.2720 - val_acc: 0.9394
# Epoch 71/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0144 - acc: 0.9953 - val_loss: 0.2766 - val_acc: 0.9402
# Epoch 72/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.0133 - acc: 0.9955 - val_loss: 0.2764 - val_acc: 0.9433
# Epoch 73/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0131 - acc: 0.9956 - val_loss: 0.2770 - val_acc: 0.9398
# Epoch 74/100
# 60000/60000 [==============================] - 27s 454us/sample - loss: 0.0125 - acc: 0.9959 - val_loss: 0.2792 - val_acc: 0.9410
# Epoch 75/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.0121 - acc: 0.9963 - val_loss: 0.2828 - val_acc: 0.9412
# Epoch 76/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.0118 - acc: 0.9964 - val_loss: 0.2851 - val_acc: 0.9405
# Epoch 77/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0114 - acc: 0.9960 - val_loss: 0.2871 - val_acc: 0.9397
# Epoch 78/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0110 - acc: 0.9965 - val_loss: 0.2923 - val_acc: 0.9404
# Epoch 79/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0110 - acc: 0.9966 - val_loss: 0.2904 - val_acc: 0.9407
# Epoch 80/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.0111 - acc: 0.9962 - val_loss: 0.2889 - val_acc: 0.9405
# Epoch 81/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0121 - acc: 0.9959 - val_loss: 0.2970 - val_acc: 0.9388
# Epoch 82/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0107 - acc: 0.9965 - val_loss: 0.2856 - val_acc: 0.9403
# Epoch 83/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0096 - acc: 0.9969 - val_loss: 0.2893 - val_acc: 0.9415
# Epoch 84/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0105 - acc: 0.9965 - val_loss: 0.2852 - val_acc: 0.9428
# Epoch 85/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.0099 - acc: 0.9970 - val_loss: 0.2967 - val_acc: 0.9415
# Epoch 86/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0099 - acc: 0.9966 - val_loss: 0.2911 - val_acc: 0.9420
# Epoch 87/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0095 - acc: 0.9970 - val_loss: 0.2918 - val_acc: 0.9425
# Epoch 88/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0104 - acc: 0.9965 - val_loss: 0.2891 - val_acc: 0.9419
# Epoch 89/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.0088 - acc: 0.9973 - val_loss: 0.2984 - val_acc: 0.9413
# Epoch 90/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0083 - acc: 0.9974 - val_loss: 0.3034 - val_acc: 0.9396
# Epoch 91/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0083 - acc: 0.9975 - val_loss: 0.2920 - val_acc: 0.9429
# Epoch 92/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0081 - acc: 0.9974 - val_loss: 0.2982 - val_acc: 0.9421
# Epoch 93/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0072 - acc: 0.9980 - val_loss: 0.2951 - val_acc: 0.9431
# Epoch 94/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0075 - acc: 0.9976 - val_loss: 0.3012 - val_acc: 0.9406
# Epoch 95/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0077 - acc: 0.9976 - val_loss: 0.2995 - val_acc: 0.9427
# Epoch 96/100
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.0079 - acc: 0.9973 - val_loss: 0.2952 - val_acc: 0.9425
# Epoch 97/100
# 60000/60000 [==============================] - 27s 454us/sample - loss: 0.0079 - acc: 0.9974 - val_loss: 0.3013 - val_acc: 0.9409
# Epoch 98/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0073 - acc: 0.9978 - val_loss: 0.3038 - val_acc: 0.9426
# Epoch 99/100
# 60000/60000 [==============================] - 27s 456us/sample - loss: 0.0071 - acc: 0.9978 - val_loss: 0.3036 - val_acc: 0.9429
# Epoch 100/100
# 60000/60000 [==============================] - 27s 455us/sample - loss: 0.0076 - acc: 0.9974 - val_loss: 0.3072 - val_acc: 0.9410
# 10000/10000 [==============================] - 1s 148us/sample - loss: 0.3072 - acc: 0.9410













