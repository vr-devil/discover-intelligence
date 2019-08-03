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

optimizer = tf.keras.optimizers.SGD()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
writer.flush()

model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

writer.flush()

# Train on 60000 samples, validate on 10000 samples
# Epoch 1/100
# 60000/60000 [==============================] - 26s 428us/sample - loss: 0.4835 - acc: 0.8250 - val_loss: 0.3443 - val_acc: 0.8750
# Epoch 2/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.3461 - acc: 0.8730 - val_loss: 0.2969 - val_acc: 0.8909
# Epoch 3/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.3048 - acc: 0.8887 - val_loss: 0.2708 - val_acc: 0.9004
# Epoch 4/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.2795 - acc: 0.8966 - val_loss: 0.2967 - val_acc: 0.8880
# Epoch 5/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.2612 - acc: 0.9038 - val_loss: 0.2529 - val_acc: 0.9088
# Epoch 6/100
# 60000/60000 [==============================] - 23s 388us/sample - loss: 0.2492 - acc: 0.9069 - val_loss: 0.2343 - val_acc: 0.9146
# Epoch 7/100
# 60000/60000 [==============================] - 23s 388us/sample - loss: 0.2361 - acc: 0.9126 - val_loss: 0.2458 - val_acc: 0.9094
# Epoch 8/100
# 60000/60000 [==============================] - 23s 388us/sample - loss: 0.2207 - acc: 0.9179 - val_loss: 0.2382 - val_acc: 0.9117
# Epoch 9/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.2123 - acc: 0.9215 - val_loss: 0.2151 - val_acc: 0.9218
# Epoch 10/100
# 60000/60000 [==============================] - 23s 389us/sample - loss: 0.2069 - acc: 0.9232 - val_loss: 0.2314 - val_acc: 0.9138
# Epoch 11/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.1963 - acc: 0.9268 - val_loss: 0.2366 - val_acc: 0.9145
# Epoch 12/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.1895 - acc: 0.9300 - val_loss: 0.2035 - val_acc: 0.9270
# Epoch 13/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.1851 - acc: 0.9304 - val_loss: 0.2333 - val_acc: 0.9122
# Epoch 14/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.1755 - acc: 0.9340 - val_loss: 0.2199 - val_acc: 0.9201
# Epoch 15/100
# 60000/60000 [==============================] - 23s 392us/sample - loss: 0.1705 - acc: 0.9373 - val_loss: 0.1970 - val_acc: 0.9302
# Epoch 16/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.1681 - acc: 0.9374 - val_loss: 0.2207 - val_acc: 0.9235
# Epoch 17/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.1619 - acc: 0.9397 - val_loss: 0.2099 - val_acc: 0.9249
# Epoch 18/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.1575 - acc: 0.9412 - val_loss: 0.1986 - val_acc: 0.9326
# Epoch 19/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.1500 - acc: 0.9432 - val_loss: 0.1902 - val_acc: 0.9329
# Epoch 20/100
# 60000/60000 [==============================] - 23s 389us/sample - loss: 0.1478 - acc: 0.9441 - val_loss: 0.1935 - val_acc: 0.9350
# Epoch 21/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.1418 - acc: 0.9463 - val_loss: 0.2016 - val_acc: 0.9312
# Epoch 22/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.1398 - acc: 0.9477 - val_loss: 0.1998 - val_acc: 0.9302
# Epoch 23/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.1333 - acc: 0.9492 - val_loss: 0.2039 - val_acc: 0.9291
# Epoch 24/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.1310 - acc: 0.9506 - val_loss: 0.1881 - val_acc: 0.9336
# Epoch 25/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.1256 - acc: 0.9527 - val_loss: 0.2174 - val_acc: 0.9267
# Epoch 26/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.1256 - acc: 0.9529 - val_loss: 0.2145 - val_acc: 0.9256
# Epoch 27/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.1201 - acc: 0.9548 - val_loss: 0.1960 - val_acc: 0.9343
# Epoch 28/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.1162 - acc: 0.9565 - val_loss: 0.1978 - val_acc: 0.9344
# Epoch 29/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.1141 - acc: 0.9571 - val_loss: 0.1992 - val_acc: 0.9340
# Epoch 30/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.1101 - acc: 0.9592 - val_loss: 0.2175 - val_acc: 0.9253
# Epoch 31/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.1078 - acc: 0.9596 - val_loss: 0.1945 - val_acc: 0.9356
# Epoch 32/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.1043 - acc: 0.9606 - val_loss: 0.1947 - val_acc: 0.9371
# Epoch 33/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.1024 - acc: 0.9625 - val_loss: 0.2035 - val_acc: 0.9348
# Epoch 34/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.0989 - acc: 0.9634 - val_loss: 0.2194 - val_acc: 0.9294
# Epoch 35/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.0944 - acc: 0.9635 - val_loss: 0.2056 - val_acc: 0.9340
# Epoch 36/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.0943 - acc: 0.9637 - val_loss: 0.2001 - val_acc: 0.9384
# Epoch 37/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0913 - acc: 0.9659 - val_loss: 0.2157 - val_acc: 0.9321
# Epoch 38/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.0918 - acc: 0.9659 - val_loss: 0.2181 - val_acc: 0.9314
# Epoch 39/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.0854 - acc: 0.9681 - val_loss: 0.2147 - val_acc: 0.9332
# Epoch 40/100
# 60000/60000 [==============================] - 24s 393us/sample - loss: 0.0846 - acc: 0.9674 - val_loss: 0.2093 - val_acc: 0.9343
# Epoch 41/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0841 - acc: 0.9697 - val_loss: 0.2137 - val_acc: 0.9318
# Epoch 42/100
# 60000/60000 [==============================] - 24s 392us/sample - loss: 0.0851 - acc: 0.9678 - val_loss: 0.2046 - val_acc: 0.9391
# Epoch 43/100
# 60000/60000 [==============================] - 23s 389us/sample - loss: 0.0772 - acc: 0.9707 - val_loss: 0.2301 - val_acc: 0.9302
# Epoch 44/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0761 - acc: 0.9710 - val_loss: 0.2140 - val_acc: 0.9362
# Epoch 45/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.0769 - acc: 0.9710 - val_loss: 0.2097 - val_acc: 0.9384
# Epoch 46/100
# 60000/60000 [==============================] - 23s 389us/sample - loss: 0.0738 - acc: 0.9720 - val_loss: 0.2201 - val_acc: 0.9316
# Epoch 47/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0708 - acc: 0.9740 - val_loss: 0.2096 - val_acc: 0.9383
# Epoch 48/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0701 - acc: 0.9732 - val_loss: 0.2059 - val_acc: 0.9406
# Epoch 49/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.0683 - acc: 0.9748 - val_loss: 0.2527 - val_acc: 0.9255
# Epoch 50/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0671 - acc: 0.9746 - val_loss: 0.2266 - val_acc: 0.9353
# Epoch 51/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0650 - acc: 0.9759 - val_loss: 0.2248 - val_acc: 0.9352
# Epoch 52/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0627 - acc: 0.9764 - val_loss: 0.2139 - val_acc: 0.9392
# Epoch 53/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0642 - acc: 0.9763 - val_loss: 0.2401 - val_acc: 0.9314
# Epoch 54/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0609 - acc: 0.9779 - val_loss: 0.2163 - val_acc: 0.9393
# Epoch 55/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.0579 - acc: 0.9782 - val_loss: 0.2295 - val_acc: 0.9388
# Epoch 56/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0564 - acc: 0.9792 - val_loss: 0.2542 - val_acc: 0.9320
# Epoch 57/100
# 60000/60000 [==============================] - 23s 389us/sample - loss: 0.0550 - acc: 0.9798 - val_loss: 0.2243 - val_acc: 0.9396
# Epoch 58/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.0568 - acc: 0.9785 - val_loss: 0.2171 - val_acc: 0.9429
# Epoch 59/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0556 - acc: 0.9795 - val_loss: 0.2259 - val_acc: 0.9396
# Epoch 60/100
# 60000/60000 [==============================] - 24s 394us/sample - loss: 0.0532 - acc: 0.9795 - val_loss: 0.2392 - val_acc: 0.9391
# Epoch 61/100
# 60000/60000 [==============================] - 24s 396us/sample - loss: 0.0509 - acc: 0.9806 - val_loss: 0.2317 - val_acc: 0.9382
# Epoch 62/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.0515 - acc: 0.9813 - val_loss: 0.2255 - val_acc: 0.9391
# Epoch 63/100
# 60000/60000 [==============================] - 24s 396us/sample - loss: 0.0475 - acc: 0.9822 - val_loss: 0.2334 - val_acc: 0.9396
# Epoch 64/100
# 60000/60000 [==============================] - 25s 414us/sample - loss: 0.0486 - acc: 0.9818 - val_loss: 0.2303 - val_acc: 0.9384
# Epoch 65/100
# 60000/60000 [==============================] - 24s 396us/sample - loss: 0.0451 - acc: 0.9829 - val_loss: 0.2363 - val_acc: 0.9389
# Epoch 66/100
# 60000/60000 [==============================] - 24s 407us/sample - loss: 0.0430 - acc: 0.9840 - val_loss: 0.2301 - val_acc: 0.9415
# Epoch 67/100
# 60000/60000 [==============================] - 24s 394us/sample - loss: 0.0442 - acc: 0.9837 - val_loss: 0.2514 - val_acc: 0.9349
# Epoch 68/100
# 60000/60000 [==============================] - 24s 393us/sample - loss: 0.0455 - acc: 0.9830 - val_loss: 0.2307 - val_acc: 0.9413
# Epoch 69/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.0426 - acc: 0.9839 - val_loss: 0.2358 - val_acc: 0.9390
# Epoch 70/100
# 60000/60000 [==============================] - 23s 392us/sample - loss: 0.0425 - acc: 0.9849 - val_loss: 0.2495 - val_acc: 0.9380
# Epoch 71/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.0410 - acc: 0.9851 - val_loss: 0.2514 - val_acc: 0.9396
# Epoch 72/100
# 60000/60000 [==============================] - 24s 395us/sample - loss: 0.0412 - acc: 0.9842 - val_loss: 0.2547 - val_acc: 0.9356
# Epoch 73/100
# 60000/60000 [==============================] - 24s 403us/sample - loss: 0.0376 - acc: 0.9861 - val_loss: 0.2475 - val_acc: 0.9391
# Epoch 74/100
# 60000/60000 [==============================] - 24s 404us/sample - loss: 0.0357 - acc: 0.9868 - val_loss: 0.3053 - val_acc: 0.9262
# Epoch 75/100
# 60000/60000 [==============================] - 24s 400us/sample - loss: 0.0380 - acc: 0.9865 - val_loss: 0.2673 - val_acc: 0.9346
# Epoch 76/100
# 60000/60000 [==============================] - 24s 398us/sample - loss: 0.0337 - acc: 0.9880 - val_loss: 0.2409 - val_acc: 0.9414
# Epoch 77/100
# 60000/60000 [==============================] - 24s 403us/sample - loss: 0.0345 - acc: 0.9876 - val_loss: 0.2529 - val_acc: 0.9394
# Epoch 78/100
# 60000/60000 [==============================] - 24s 392us/sample - loss: 0.0333 - acc: 0.9876 - val_loss: 0.2670 - val_acc: 0.9375
# Epoch 79/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0332 - acc: 0.9880 - val_loss: 0.2574 - val_acc: 0.9386
# Epoch 80/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.0351 - acc: 0.9870 - val_loss: 0.2570 - val_acc: 0.9397
# Epoch 81/100
# 60000/60000 [==============================] - 24s 393us/sample - loss: 0.0311 - acc: 0.9883 - val_loss: 0.2621 - val_acc: 0.9390
# Epoch 82/100
# 60000/60000 [==============================] - 24s 395us/sample - loss: 0.0321 - acc: 0.9886 - val_loss: 0.2710 - val_acc: 0.9339
# Epoch 83/100
# 60000/60000 [==============================] - 24s 402us/sample - loss: 0.0315 - acc: 0.9884 - val_loss: 0.2538 - val_acc: 0.9410
# Epoch 84/100
# 60000/60000 [==============================] - 24s 397us/sample - loss: 0.0318 - acc: 0.9884 - val_loss: 0.2535 - val_acc: 0.9395
# Epoch 85/100
# 60000/60000 [==============================] - 24s 394us/sample - loss: 0.0290 - acc: 0.9891 - val_loss: 0.2768 - val_acc: 0.9363
# Epoch 86/100
# 60000/60000 [==============================] - 23s 392us/sample - loss: 0.0290 - acc: 0.9894 - val_loss: 0.2688 - val_acc: 0.9401
# Epoch 87/100
# 60000/60000 [==============================] - 23s 389us/sample - loss: 0.0265 - acc: 0.9903 - val_loss: 0.2628 - val_acc: 0.9400
# Epoch 88/100
# 60000/60000 [==============================] - 24s 400us/sample - loss: 0.0260 - acc: 0.9906 - val_loss: 0.2682 - val_acc: 0.9406
# Epoch 89/100
# 60000/60000 [==============================] - 24s 393us/sample - loss: 0.0273 - acc: 0.9895 - val_loss: 0.2688 - val_acc: 0.9398
# Epoch 90/100
# 60000/60000 [==============================] - 24s 392us/sample - loss: 0.0276 - acc: 0.9901 - val_loss: 0.2676 - val_acc: 0.9415
# Epoch 91/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0250 - acc: 0.9912 - val_loss: 0.2832 - val_acc: 0.9378
# Epoch 92/100
# 60000/60000 [==============================] - 23s 388us/sample - loss: 0.0253 - acc: 0.9905 - val_loss: 0.2569 - val_acc: 0.9420
# Epoch 93/100
# 60000/60000 [==============================] - 24s 392us/sample - loss: 0.0293 - acc: 0.9891 - val_loss: 0.2774 - val_acc: 0.9359
# Epoch 94/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0251 - acc: 0.9910 - val_loss: 0.2790 - val_acc: 0.9379
# Epoch 95/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0233 - acc: 0.9919 - val_loss: 0.2634 - val_acc: 0.9404
# Epoch 96/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.0246 - acc: 0.9909 - val_loss: 0.2771 - val_acc: 0.9395
# Epoch 97/100
# 60000/60000 [==============================] - 24s 394us/sample - loss: 0.0244 - acc: 0.9909 - val_loss: 0.3221 - val_acc: 0.9305
# Epoch 98/100
# 60000/60000 [==============================] - 24s 392us/sample - loss: 0.0230 - acc: 0.9915 - val_loss: 0.2812 - val_acc: 0.9388
# Epoch 99/100
# 60000/60000 [==============================] - 23s 390us/sample - loss: 0.0225 - acc: 0.9923 - val_loss: 0.2861 - val_acc: 0.9378
# Epoch 100/100
# 60000/60000 [==============================] - 23s 391us/sample - loss: 0.0241 - acc: 0.9916 - val_loss: 0.2988 - val_acc: 0.9360
# 10000/10000 [==============================] - 2s 152us/sample - loss: 0.2988 - acc: 0.9360












