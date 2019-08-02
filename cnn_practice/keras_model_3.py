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
    layers.BatchNormalization(),
    layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.BatchNormalization(),
    layers.Dense(2048, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(2048, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.SGD(0.05)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

# Train on 60000 samples, validate on 10000 samples
# Epoch 1/100
# 60000/60000 [==============================] - 43s 708us/sample - loss: 0.5344 - acc: 0.8418 - val_loss: 0.5611 - val_acc: 0.8272
# Epoch 2/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 0.2934 - acc: 0.8968 - val_loss: 0.3310 - val_acc: 0.8839
# Epoch 3/100
# 60000/60000 [==============================] - 39s 658us/sample - loss: 0.2361 - acc: 0.9144 - val_loss: 0.2491 - val_acc: 0.9149
# Epoch 4/100
# 60000/60000 [==============================] - 39s 654us/sample - loss: 0.1957 - acc: 0.9287 - val_loss: 0.2701 - val_acc: 0.9092
# Epoch 5/100
# 60000/60000 [==============================] - 39s 649us/sample - loss: 0.1645 - acc: 0.9398 - val_loss: 0.2946 - val_acc: 0.9012
# Epoch 6/100
# 60000/60000 [==============================] - 39s 646us/sample - loss: 0.1439 - acc: 0.9474 - val_loss: 0.2732 - val_acc: 0.9141
# Epoch 7/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 0.1154 - acc: 0.9578 - val_loss: 0.3450 - val_acc: 0.9081
# Epoch 8/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 0.0971 - acc: 0.9638 - val_loss: 0.2732 - val_acc: 0.9287
# Epoch 9/100
# 60000/60000 [==============================] - 39s 645us/sample - loss: 0.0822 - acc: 0.9696 - val_loss: 0.2981 - val_acc: 0.9217
# Epoch 10/100
# 60000/60000 [==============================] - 39s 646us/sample - loss: 0.0739 - acc: 0.9723 - val_loss: 0.3045 - val_acc: 0.9223
# Epoch 11/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 0.0571 - acc: 0.9788 - val_loss: 0.2870 - val_acc: 0.9323
# Epoch 12/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 0.0498 - acc: 0.9815 - val_loss: 0.3354 - val_acc: 0.9239
# Epoch 13/100
# 60000/60000 [==============================] - 39s 647us/sample - loss: 0.0437 - acc: 0.9838 - val_loss: 0.3350 - val_acc: 0.9276
# Epoch 14/100
# 60000/60000 [==============================] - 39s 645us/sample - loss: 0.0408 - acc: 0.9852 - val_loss: 0.3433 - val_acc: 0.9247
# Epoch 15/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 0.0327 - acc: 0.9880 - val_loss: 0.3276 - val_acc: 0.9309
# Epoch 16/100
# 60000/60000 [==============================] - 39s 645us/sample - loss: 0.0231 - acc: 0.9916 - val_loss: 0.3769 - val_acc: 0.9307
# Epoch 17/100
# 60000/60000 [==============================] - 39s 646us/sample - loss: 0.0238 - acc: 0.9916 - val_loss: 0.3611 - val_acc: 0.9329
# Epoch 18/100
# 60000/60000 [==============================] - 39s 645us/sample - loss: 0.0244 - acc: 0.9912 - val_loss: 0.3738 - val_acc: 0.9302
# Epoch 19/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 0.0177 - acc: 0.9938 - val_loss: 0.3732 - val_acc: 0.9312
# Epoch 20/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 0.0122 - acc: 0.9959 - val_loss: 0.4000 - val_acc: 0.9329
# Epoch 21/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 0.0165 - acc: 0.9945 - val_loss: 0.4051 - val_acc: 0.9322
# Epoch 22/100
# 60000/60000 [==============================] - 39s 649us/sample - loss: 0.0145 - acc: 0.9948 - val_loss: 0.4482 - val_acc: 0.9278
# Epoch 23/100
# 60000/60000 [==============================] - 40s 672us/sample - loss: 0.0138 - acc: 0.9953 - val_loss: 0.4193 - val_acc: 0.9271
# Epoch 24/100
# 60000/60000 [==============================] - 41s 690us/sample - loss: 0.0089 - acc: 0.9970 - val_loss: 0.4086 - val_acc: 0.9344
# Epoch 25/100
# 60000/60000 [==============================] - 41s 686us/sample - loss: 0.0078 - acc: 0.9973 - val_loss: 0.4524 - val_acc: 0.9287
# Epoch 26/100
# 60000/60000 [==============================] - 43s 722us/sample - loss: 0.0074 - acc: 0.9976 - val_loss: 0.4130 - val_acc: 0.9343
# Epoch 27/100
# 60000/60000 [==============================] - 42s 702us/sample - loss: 0.0072 - acc: 0.9973 - val_loss: 0.4857 - val_acc: 0.9264
# Epoch 28/100
# 60000/60000 [==============================] - 41s 690us/sample - loss: 0.0072 - acc: 0.9975 - val_loss: 0.4368 - val_acc: 0.9335
# Epoch 29/100
# 60000/60000 [==============================] - 41s 678us/sample - loss: 0.0097 - acc: 0.9969 - val_loss: 0.4273 - val_acc: 0.9303
# Epoch 30/100
# 60000/60000 [==============================] - 41s 680us/sample - loss: 0.0052 - acc: 0.9983 - val_loss: 0.4336 - val_acc: 0.9344
# Epoch 31/100
# 60000/60000 [==============================] - 42s 695us/sample - loss: 0.0050 - acc: 0.9984 - val_loss: 0.4763 - val_acc: 0.9308
# Epoch 32/100
# 60000/60000 [==============================] - 42s 697us/sample - loss: 0.0062 - acc: 0.9980 - val_loss: 0.4317 - val_acc: 0.9364
# Epoch 33/100
# 60000/60000 [==============================] - 42s 693us/sample - loss: 0.0042 - acc: 0.9987 - val_loss: 0.4386 - val_acc: 0.9355
# Epoch 34/100
# 60000/60000 [==============================] - 41s 688us/sample - loss: 0.0039 - acc: 0.9987 - val_loss: 0.4527 - val_acc: 0.9332
# Epoch 35/100
# 60000/60000 [==============================] - 41s 682us/sample - loss: 0.0021 - acc: 0.9994 - val_loss: 0.4694 - val_acc: 0.9350
# Epoch 36/100
# 60000/60000 [==============================] - 42s 693us/sample - loss: 0.0024 - acc: 0.9993 - val_loss: 0.4816 - val_acc: 0.9355
# Epoch 37/100
# 60000/60000 [==============================] - 41s 677us/sample - loss: 0.0049 - acc: 0.9984 - val_loss: 0.4489 - val_acc: 0.9328
# Epoch 38/100
# 60000/60000 [==============================] - 39s 658us/sample - loss: 0.0040 - acc: 0.9988 - val_loss: 0.4530 - val_acc: 0.9312
# Epoch 39/100
# 60000/60000 [==============================] - 38s 630us/sample - loss: 0.0029 - acc: 0.9991 - val_loss: 0.4561 - val_acc: 0.9366
# Epoch 40/100
# 60000/60000 [==============================] - 37s 625us/sample - loss: 0.0019 - acc: 0.9995 - val_loss: 0.4645 - val_acc: 0.9340
# Epoch 41/100
# 60000/60000 [==============================] - 38s 626us/sample - loss: 0.0020 - acc: 0.9995 - val_loss: 0.4707 - val_acc: 0.9357
# Epoch 42/100
# 60000/60000 [==============================] - 38s 626us/sample - loss: 0.0017 - acc: 0.9996 - val_loss: 0.4800 - val_acc: 0.9326
# Epoch 43/100
# 60000/60000 [==============================] - 37s 625us/sample - loss: 8.3761e-04 - acc: 0.9998 - val_loss: 0.4710 - val_acc: 0.9356
# Epoch 44/100
# 60000/60000 [==============================] - 38s 625us/sample - loss: 0.0012 - acc: 0.9997 - val_loss: 0.4791 - val_acc: 0.9360
# Epoch 45/100
# 60000/60000 [==============================] - 38s 639us/sample - loss: 8.7904e-04 - acc: 0.9997 - val_loss: 0.4745 - val_acc: 0.9372
# Epoch 46/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 3.2425e-04 - acc: 0.9999 - val_loss: 0.4724 - val_acc: 0.9360
# Epoch 47/100
# 60000/60000 [==============================] - 39s 650us/sample - loss: 5.8314e-04 - acc: 0.9998 - val_loss: 0.4730 - val_acc: 0.9381
# Epoch 48/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 1.5875e-04 - acc: 1.0000 - val_loss: 0.4765 - val_acc: 0.9389
# Epoch 49/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 9.2940e-05 - acc: 1.0000 - val_loss: 0.4729 - val_acc: 0.9387
# Epoch 50/100
# 60000/60000 [==============================] - 39s 649us/sample - loss: 7.3618e-05 - acc: 1.0000 - val_loss: 0.4776 - val_acc: 0.9383
# Epoch 51/100
# 60000/60000 [==============================] - 39s 646us/sample - loss: 6.6549e-05 - acc: 1.0000 - val_loss: 0.4777 - val_acc: 0.9389
# Epoch 52/100
# 60000/60000 [==============================] - 39s 643us/sample - loss: 7.5822e-05 - acc: 1.0000 - val_loss: 0.4926 - val_acc: 0.9390
# Epoch 53/100
# 60000/60000 [==============================] - 39s 645us/sample - loss: 1.1488e-04 - acc: 1.0000 - val_loss: 0.4914 - val_acc: 0.9389
# Epoch 54/100
# 60000/60000 [==============================] - 39s 642us/sample - loss: 6.7448e-05 - acc: 1.0000 - val_loss: 0.4911 - val_acc: 0.9394
# Epoch 55/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 6.7308e-05 - acc: 1.0000 - val_loss: 0.4937 - val_acc: 0.9384
# Epoch 56/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 5.0452e-05 - acc: 1.0000 - val_loss: 0.4918 - val_acc: 0.9390
# Epoch 57/100
# 60000/60000 [==============================] - 39s 645us/sample - loss: 5.0662e-05 - acc: 1.0000 - val_loss: 0.4882 - val_acc: 0.9387
# Epoch 58/100
# 60000/60000 [==============================] - 39s 645us/sample - loss: 7.8988e-05 - acc: 1.0000 - val_loss: 0.4877 - val_acc: 0.9385
# Epoch 59/100
# 60000/60000 [==============================] - 39s 643us/sample - loss: 4.0658e-05 - acc: 1.0000 - val_loss: 0.4924 - val_acc: 0.9386
# Epoch 60/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 4.6492e-05 - acc: 1.0000 - val_loss: 0.4906 - val_acc: 0.9388
# Epoch 61/100
# 60000/60000 [==============================] - 39s 649us/sample - loss: 3.6782e-05 - acc: 1.0000 - val_loss: 0.4888 - val_acc: 0.9396
# Epoch 62/100
# 60000/60000 [==============================] - 39s 650us/sample - loss: 3.3213e-05 - acc: 1.0000 - val_loss: 0.4909 - val_acc: 0.9394
# Epoch 63/100
# 60000/60000 [==============================] - 39s 649us/sample - loss: 2.8392e-05 - acc: 1.0000 - val_loss: 0.4910 - val_acc: 0.9387
# Epoch 64/100
# 60000/60000 [==============================] - 39s 643us/sample - loss: 2.9785e-05 - acc: 1.0000 - val_loss: 0.4904 - val_acc: 0.9395
# Epoch 65/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 2.2249e-05 - acc: 1.0000 - val_loss: 0.4943 - val_acc: 0.9391
# Epoch 66/100
# 60000/60000 [==============================] - 39s 645us/sample - loss: 1.9375e-05 - acc: 1.0000 - val_loss: 0.4933 - val_acc: 0.9395
# Epoch 67/100
# 60000/60000 [==============================] - 39s 643us/sample - loss: 4.8247e-05 - acc: 1.0000 - val_loss: 0.4954 - val_acc: 0.9392
# Epoch 68/100
# 60000/60000 [==============================] - 39s 643us/sample - loss: 7.4582e-05 - acc: 1.0000 - val_loss: 0.4919 - val_acc: 0.9387
# Epoch 69/100
# 60000/60000 [==============================] - 39s 645us/sample - loss: 5.6721e-05 - acc: 1.0000 - val_loss: 0.4915 - val_acc: 0.9381
# Epoch 70/100
# 60000/60000 [==============================] - 39s 643us/sample - loss: 2.6887e-04 - acc: 0.9999 - val_loss: 0.4977 - val_acc: 0.9385
# Epoch 71/100
# 60000/60000 [==============================] - 39s 646us/sample - loss: 8.0357e-05 - acc: 1.0000 - val_loss: 0.4995 - val_acc: 0.9375
# Epoch 72/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 5.4637e-05 - acc: 1.0000 - val_loss: 0.4943 - val_acc: 0.9378
# Epoch 73/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 2.6223e-05 - acc: 1.0000 - val_loss: 0.4962 - val_acc: 0.9376
# Epoch 74/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 3.7510e-05 - acc: 1.0000 - val_loss: 0.4953 - val_acc: 0.9383
# Epoch 75/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 2.0421e-05 - acc: 1.0000 - val_loss: 0.4968 - val_acc: 0.9376
# Epoch 76/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 1.8682e-05 - acc: 1.0000 - val_loss: 0.5000 - val_acc: 0.9374
# Epoch 77/100
# 60000/60000 [==============================] - 39s 645us/sample - loss: 1.9575e-05 - acc: 1.0000 - val_loss: 0.5032 - val_acc: 0.9379
# Epoch 78/100
# 60000/60000 [==============================] - 39s 643us/sample - loss: 2.7749e-05 - acc: 1.0000 - val_loss: 0.5012 - val_acc: 0.9374
# Epoch 79/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 3.3922e-05 - acc: 1.0000 - val_loss: 0.5057 - val_acc: 0.9370
# Epoch 80/100
# 60000/60000 [==============================] - 39s 648us/sample - loss: 3.5113e-05 - acc: 1.0000 - val_loss: 0.5062 - val_acc: 0.9371
# Epoch 81/100
# 60000/60000 [==============================] - 39s 648us/sample - loss: 4.4569e-05 - acc: 1.0000 - val_loss: 0.5023 - val_acc: 0.9380
# Epoch 82/100
# 60000/60000 [==============================] - 39s 645us/sample - loss: 2.0348e-05 - acc: 1.0000 - val_loss: 0.5031 - val_acc: 0.9373
# Epoch 83/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 1.7360e-05 - acc: 1.0000 - val_loss: 0.5005 - val_acc: 0.9373
# Epoch 84/100
# 60000/60000 [==============================] - 39s 644us/sample - loss: 1.9726e-05 - acc: 1.0000 - val_loss: 0.5009 - val_acc: 0.9373
# Epoch 85/100
# 60000/60000 [==============================] - 39s 643us/sample - loss: 2.1958e-05 - acc: 1.0000 - val_loss: 0.5031 - val_acc: 0.9375
# Epoch 86/100
# 60000/60000 [==============================] - 39s 654us/sample - loss: 5.3677e-05 - acc: 1.0000 - val_loss: 0.5037 - val_acc: 0.9376
# Epoch 87/100
# 60000/60000 [==============================] - 39s 651us/sample - loss: 1.7323e-05 - acc: 1.0000 - val_loss: 0.5066 - val_acc: 0.9382
# Epoch 88/100
# 60000/60000 [==============================] - 39s 653us/sample - loss: 1.3207e-05 - acc: 1.0000 - val_loss: 0.5071 - val_acc: 0.9379
# Epoch 89/100
# 60000/60000 [==============================] - 39s 657us/sample - loss: 1.6993e-05 - acc: 1.0000 - val_loss: 0.5080 - val_acc: 0.9385
# Epoch 90/100
# 60000/60000 [==============================] - 39s 652us/sample - loss: 1.1458e-05 - acc: 1.0000 - val_loss: 0.5069 - val_acc: 0.9380
# Epoch 91/100
# 60000/60000 [==============================] - 39s 648us/sample - loss: 1.4443e-05 - acc: 1.0000 - val_loss: 0.5090 - val_acc: 0.9376
# Epoch 92/100
# 60000/60000 [==============================] - 38s 633us/sample - loss: 3.0607e-05 - acc: 1.0000 - val_loss: 0.5100 - val_acc: 0.9375
# Epoch 93/100
# 60000/60000 [==============================] - 38s 627us/sample - loss: 1.6840e-05 - acc: 1.0000 - val_loss: 0.5059 - val_acc: 0.9382
# Epoch 94/100
# 60000/60000 [==============================] - 38s 626us/sample - loss: 1.4413e-05 - acc: 1.0000 - val_loss: 0.5071 - val_acc: 0.9386
# Epoch 95/100
# 60000/60000 [==============================] - 38s 626us/sample - loss: 1.8216e-05 - acc: 1.0000 - val_loss: 0.5073 - val_acc: 0.9377
# Epoch 96/100
# 60000/60000 [==============================] - 38s 626us/sample - loss: 1.6115e-05 - acc: 1.0000 - val_loss: 0.5085 - val_acc: 0.9379
# Epoch 97/100
# 60000/60000 [==============================] - 38s 626us/sample - loss: 1.5452e-05 - acc: 1.0000 - val_loss: 0.5082 - val_acc: 0.9384
# Epoch 98/100
# 60000/60000 [==============================] - 38s 626us/sample - loss: 1.4230e-05 - acc: 1.0000 - val_loss: 0.5102 - val_acc: 0.9376
# Epoch 99/100
# 60000/60000 [==============================] - 38s 626us/sample - loss: 2.9870e-05 - acc: 1.0000 - val_loss: 0.5114 - val_acc: 0.9381
# Epoch 100/100
# 60000/60000 [==============================] - 37s 625us/sample - loss: 1.2559e-05 - acc: 1.0000 - val_loss: 0.5089 - val_acc: 0.9386
# 10000/10000 [==============================] - 2s 205us/sample - loss: 0.5089 - acc: 0.9386









