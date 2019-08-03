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
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (2, 2), activation='relu', padding='same'),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.4),
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
# 60000/60000 [==============================] - 28s 474us/sample - loss: 0.4527 - acc: 0.8359 - val_loss: 0.3263 - val_acc: 0.8808
# Epoch 2/100
# 60000/60000 [==============================] - 20s 325us/sample - loss: 0.3211 - acc: 0.8824 - val_loss: 0.2787 - val_acc: 0.8985
# Epoch 3/100
# 60000/60000 [==============================] - 19s 325us/sample - loss: 0.2813 - acc: 0.8964 - val_loss: 0.2938 - val_acc: 0.8916
# Epoch 4/100
# 60000/60000 [==============================] - 20s 326us/sample - loss: 0.2542 - acc: 0.9066 - val_loss: 0.2434 - val_acc: 0.9115
# Epoch 5/100
# 60000/60000 [==============================] - 20s 326us/sample - loss: 0.2353 - acc: 0.9129 - val_loss: 0.2652 - val_acc: 0.8992
# Epoch 6/100
# 60000/60000 [==============================] - 20s 327us/sample - loss: 0.2217 - acc: 0.9185 - val_loss: 0.2468 - val_acc: 0.9128
# Epoch 7/100
# 60000/60000 [==============================] - 20s 326us/sample - loss: 0.2122 - acc: 0.9216 - val_loss: 0.2241 - val_acc: 0.9196
# Epoch 8/100
# 60000/60000 [==============================] - 20s 326us/sample - loss: 0.1991 - acc: 0.9265 - val_loss: 0.2260 - val_acc: 0.9191
# Epoch 9/100
# 60000/60000 [==============================] - 20s 327us/sample - loss: 0.1886 - acc: 0.9294 - val_loss: 0.2171 - val_acc: 0.9212
# Epoch 10/100
# 60000/60000 [==============================] - 20s 327us/sample - loss: 0.1778 - acc: 0.9329 - val_loss: 0.2150 - val_acc: 0.9208
# Epoch 11/100
# 60000/60000 [==============================] - 19s 324us/sample - loss: 0.1713 - acc: 0.9366 - val_loss: 0.2196 - val_acc: 0.9243
# Epoch 12/100
# 60000/60000 [==============================] - 20s 327us/sample - loss: 0.1637 - acc: 0.9383 - val_loss: 0.2111 - val_acc: 0.9233
# Epoch 13/100
# 60000/60000 [==============================] - 20s 326us/sample - loss: 0.1565 - acc: 0.9415 - val_loss: 0.2117 - val_acc: 0.9247
# Epoch 14/100
# 60000/60000 [==============================] - 20s 326us/sample - loss: 0.1497 - acc: 0.9436 - val_loss: 0.2060 - val_acc: 0.9278
# Epoch 15/100
# 60000/60000 [==============================] - 19s 325us/sample - loss: 0.1402 - acc: 0.9482 - val_loss: 0.2059 - val_acc: 0.9283
# Epoch 16/100
# 60000/60000 [==============================] - 20s 326us/sample - loss: 0.1356 - acc: 0.9492 - val_loss: 0.2076 - val_acc: 0.9274
# Epoch 17/100
# 60000/60000 [==============================] - 20s 327us/sample - loss: 0.1295 - acc: 0.9517 - val_loss: 0.2082 - val_acc: 0.9293
# Epoch 18/100
# 60000/60000 [==============================] - 20s 325us/sample - loss: 0.1253 - acc: 0.9527 - val_loss: 0.2043 - val_acc: 0.9296
# Epoch 19/100
# 60000/60000 [==============================] - 20s 328us/sample - loss: 0.1178 - acc: 0.9554 - val_loss: 0.2034 - val_acc: 0.9300
# Epoch 20/100
# 60000/60000 [==============================] - 20s 327us/sample - loss: 0.1112 - acc: 0.9588 - val_loss: 0.2120 - val_acc: 0.9287
# Epoch 21/100
# 60000/60000 [==============================] - 19s 324us/sample - loss: 0.1066 - acc: 0.9600 - val_loss: 0.2094 - val_acc: 0.9296
# Epoch 22/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.1008 - acc: 0.9614 - val_loss: 0.2258 - val_acc: 0.9288
# Epoch 23/100
# 60000/60000 [==============================] - 19s 322us/sample - loss: 0.0978 - acc: 0.9636 - val_loss: 0.2246 - val_acc: 0.9258
# Epoch 24/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0903 - acc: 0.9659 - val_loss: 0.2164 - val_acc: 0.9287
# Epoch 25/100
# 60000/60000 [==============================] - 19s 322us/sample - loss: 0.0892 - acc: 0.9669 - val_loss: 0.2252 - val_acc: 0.9280
# Epoch 26/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0832 - acc: 0.9686 - val_loss: 0.2243 - val_acc: 0.9297
# Epoch 27/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0797 - acc: 0.9706 - val_loss: 0.2235 - val_acc: 0.9331
# Epoch 28/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0727 - acc: 0.9729 - val_loss: 0.2345 - val_acc: 0.9302
# Epoch 29/100
# 60000/60000 [==============================] - 19s 316us/sample - loss: 0.0691 - acc: 0.9743 - val_loss: 0.2480 - val_acc: 0.9267
# Epoch 30/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0678 - acc: 0.9750 - val_loss: 0.2170 - val_acc: 0.9367
# Epoch 31/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0667 - acc: 0.9758 - val_loss: 0.2443 - val_acc: 0.9280
# Epoch 32/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0605 - acc: 0.9774 - val_loss: 0.2444 - val_acc: 0.9288
# Epoch 33/100
# 60000/60000 [==============================] - 19s 322us/sample - loss: 0.0582 - acc: 0.9786 - val_loss: 0.2407 - val_acc: 0.9321
# Epoch 34/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0533 - acc: 0.9803 - val_loss: 0.2461 - val_acc: 0.9331
# Epoch 35/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0500 - acc: 0.9814 - val_loss: 0.2416 - val_acc: 0.9356
# Epoch 36/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0536 - acc: 0.9805 - val_loss: 0.2897 - val_acc: 0.9200
# Epoch 37/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0471 - acc: 0.9830 - val_loss: 0.2682 - val_acc: 0.9276
# Epoch 38/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0456 - acc: 0.9834 - val_loss: 0.2384 - val_acc: 0.9350
# Epoch 39/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0427 - acc: 0.9848 - val_loss: 0.2526 - val_acc: 0.9337
# Epoch 40/100
# 60000/60000 [==============================] - 19s 322us/sample - loss: 0.0410 - acc: 0.9857 - val_loss: 0.2628 - val_acc: 0.9332
# Epoch 41/100
# 60000/60000 [==============================] - 20s 325us/sample - loss: 0.0382 - acc: 0.9866 - val_loss: 0.2610 - val_acc: 0.9345
# Epoch 42/100
# 60000/60000 [==============================] - 19s 322us/sample - loss: 0.0366 - acc: 0.9865 - val_loss: 0.2505 - val_acc: 0.9371
# Epoch 43/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0362 - acc: 0.9870 - val_loss: 0.2494 - val_acc: 0.9356
# Epoch 44/100
# 60000/60000 [==============================] - 19s 322us/sample - loss: 0.0339 - acc: 0.9875 - val_loss: 0.2614 - val_acc: 0.9373
# Epoch 45/100
# 60000/60000 [==============================] - 19s 322us/sample - loss: 0.0323 - acc: 0.9888 - val_loss: 0.2644 - val_acc: 0.9369
# Epoch 46/100
# 60000/60000 [==============================] - 19s 319us/sample - loss: 0.0330 - acc: 0.9880 - val_loss: 0.2688 - val_acc: 0.9337
# Epoch 47/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0312 - acc: 0.9887 - val_loss: 0.2560 - val_acc: 0.9372
# Epoch 48/100
# 60000/60000 [==============================] - 19s 324us/sample - loss: 0.0297 - acc: 0.9894 - val_loss: 0.2609 - val_acc: 0.9368
# Epoch 49/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0254 - acc: 0.9915 - val_loss: 0.2677 - val_acc: 0.9371
# Epoch 50/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0261 - acc: 0.9908 - val_loss: 0.2658 - val_acc: 0.9375
# Epoch 51/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0241 - acc: 0.9913 - val_loss: 0.2712 - val_acc: 0.9359
# Epoch 52/100
# 60000/60000 [==============================] - 19s 322us/sample - loss: 0.0235 - acc: 0.9924 - val_loss: 0.3235 - val_acc: 0.9280
# Epoch 53/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0232 - acc: 0.9916 - val_loss: 0.2774 - val_acc: 0.9385
# Epoch 54/100
# 60000/60000 [==============================] - 20s 325us/sample - loss: 0.0213 - acc: 0.9927 - val_loss: 0.2930 - val_acc: 0.9357
# Epoch 55/100
# 60000/60000 [==============================] - 19s 322us/sample - loss: 0.0221 - acc: 0.9920 - val_loss: 0.2883 - val_acc: 0.9352
# Epoch 56/100
# 60000/60000 [==============================] - 19s 322us/sample - loss: 0.0206 - acc: 0.9929 - val_loss: 0.2833 - val_acc: 0.9341
# Epoch 57/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0200 - acc: 0.9929 - val_loss: 0.2844 - val_acc: 0.9369
# Epoch 58/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0195 - acc: 0.9935 - val_loss: 0.2917 - val_acc: 0.9370
# Epoch 59/100
# 60000/60000 [==============================] - 19s 323us/sample - loss: 0.0185 - acc: 0.9939 - val_loss: 0.3111 - val_acc: 0.9318
# Epoch 60/100
# 60000/60000 [==============================] - 20s 325us/sample - loss: 0.0180 - acc: 0.9938 - val_loss: 0.3020 - val_acc: 0.9354
# Epoch 61/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0161 - acc: 0.9946 - val_loss: 0.3205 - val_acc: 0.9332
# Epoch 62/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0164 - acc: 0.9945 - val_loss: 0.3080 - val_acc: 0.9359
# Epoch 63/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0167 - acc: 0.9943 - val_loss: 0.3092 - val_acc: 0.9340
# Epoch 64/100
# 60000/60000 [==============================] - 19s 319us/sample - loss: 0.0165 - acc: 0.9945 - val_loss: 0.2962 - val_acc: 0.9372
# Epoch 65/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0154 - acc: 0.9949 - val_loss: 0.2973 - val_acc: 0.9392
# Epoch 66/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0151 - acc: 0.9951 - val_loss: 0.3157 - val_acc: 0.9365
# Epoch 67/100
# 60000/60000 [==============================] - 19s 322us/sample - loss: 0.0146 - acc: 0.9949 - val_loss: 0.2966 - val_acc: 0.9398
# Epoch 68/100
# 60000/60000 [==============================] - 19s 323us/sample - loss: 0.0150 - acc: 0.9949 - val_loss: 0.3093 - val_acc: 0.9368
# Epoch 69/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0132 - acc: 0.9958 - val_loss: 0.3057 - val_acc: 0.9372
# Epoch 70/100
# 60000/60000 [==============================] - 19s 322us/sample - loss: 0.0135 - acc: 0.9954 - val_loss: 0.3168 - val_acc: 0.9372
# Epoch 71/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0126 - acc: 0.9959 - val_loss: 0.3021 - val_acc: 0.9388
# Epoch 72/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0127 - acc: 0.9958 - val_loss: 0.3126 - val_acc: 0.9367
# Epoch 73/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0114 - acc: 0.9965 - val_loss: 0.3099 - val_acc: 0.9396
# Epoch 74/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0104 - acc: 0.9969 - val_loss: 0.3097 - val_acc: 0.9377
# Epoch 75/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0105 - acc: 0.9967 - val_loss: 0.3132 - val_acc: 0.9389
# Epoch 76/100
# 60000/60000 [==============================] - 19s 324us/sample - loss: 0.0122 - acc: 0.9958 - val_loss: 0.3240 - val_acc: 0.9368
# Epoch 77/100
# 60000/60000 [==============================] - 20s 328us/sample - loss: 0.0113 - acc: 0.9962 - val_loss: 0.3128 - val_acc: 0.9393
# Epoch 78/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0109 - acc: 0.9965 - val_loss: 0.3181 - val_acc: 0.9393
# Epoch 79/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0109 - acc: 0.9962 - val_loss: 0.3112 - val_acc: 0.9390
# Epoch 80/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0124 - acc: 0.9960 - val_loss: 0.3196 - val_acc: 0.9363
# Epoch 81/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0093 - acc: 0.9969 - val_loss: 0.3121 - val_acc: 0.9383
# Epoch 82/100
# 60000/60000 [==============================] - 19s 324us/sample - loss: 0.0110 - acc: 0.9962 - val_loss: 0.3301 - val_acc: 0.9380
# Epoch 83/100
# 60000/60000 [==============================] - 20s 325us/sample - loss: 0.0107 - acc: 0.9963 - val_loss: 0.3299 - val_acc: 0.9384
# Epoch 84/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0089 - acc: 0.9972 - val_loss: 0.3196 - val_acc: 0.9401
# Epoch 85/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0089 - acc: 0.9972 - val_loss: 0.3379 - val_acc: 0.9370
# Epoch 86/100
# 60000/60000 [==============================] - 20s 330us/sample - loss: 0.0085 - acc: 0.9973 - val_loss: 0.3437 - val_acc: 0.9371
# Epoch 87/100
# 60000/60000 [==============================] - 19s 324us/sample - loss: 0.0101 - acc: 0.9968 - val_loss: 0.3246 - val_acc: 0.9401
# Epoch 88/100
# 60000/60000 [==============================] - 19s 324us/sample - loss: 0.0091 - acc: 0.9969 - val_loss: 0.3189 - val_acc: 0.9379
# Epoch 89/100
# 60000/60000 [==============================] - 19s 324us/sample - loss: 0.0080 - acc: 0.9974 - val_loss: 0.3295 - val_acc: 0.9384
# Epoch 90/100
# 60000/60000 [==============================] - 19s 322us/sample - loss: 0.0090 - acc: 0.9969 - val_loss: 0.3438 - val_acc: 0.9368
# Epoch 91/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0088 - acc: 0.9972 - val_loss: 0.3227 - val_acc: 0.9376
# Epoch 92/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0084 - acc: 0.9972 - val_loss: 0.3457 - val_acc: 0.9368
# Epoch 93/100
# 60000/60000 [==============================] - 19s 321us/sample - loss: 0.0089 - acc: 0.9971 - val_loss: 0.3238 - val_acc: 0.9376
# Epoch 94/100
# 60000/60000 [==============================] - 19s 322us/sample - loss: 0.0085 - acc: 0.9972 - val_loss: 0.3510 - val_acc: 0.9361
# Epoch 95/100
# 60000/60000 [==============================] - 19s 322us/sample - loss: 0.0078 - acc: 0.9974 - val_loss: 0.3302 - val_acc: 0.9397
# Epoch 96/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0082 - acc: 0.9973 - val_loss: 0.3392 - val_acc: 0.9371
# Epoch 97/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0076 - acc: 0.9977 - val_loss: 0.3347 - val_acc: 0.9392
# Epoch 98/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0082 - acc: 0.9973 - val_loss: 0.3169 - val_acc: 0.9398
# Epoch 99/100
# 60000/60000 [==============================] - 19s 319us/sample - loss: 0.0079 - acc: 0.9974 - val_loss: 0.3326 - val_acc: 0.9398
# Epoch 100/100
# 60000/60000 [==============================] - 19s 320us/sample - loss: 0.0075 - acc: 0.9975 - val_loss: 0.3449 - val_acc: 0.9375
# 10000/10000 [==============================] - 1s 132us/sample - loss: 0.3449 - acc: 0.9375











