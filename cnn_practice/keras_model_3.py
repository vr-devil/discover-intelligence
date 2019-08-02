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
    layers.Dropout(0.4),
    layers.Dense(1024, activation='relu'),
    layers.Dense(1024, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

# Train on 60000 samples, validate on 10000 samples
# Epoch 1/100
# 60000/60000 [==============================] - 35s 588us/sample - loss: 0.4466 - acc: 0.8373 - val_loss: 0.3324 - val_acc: 0.8777
# Epoch 2/100
# 60000/60000 [==============================] - 33s 552us/sample - loss: 0.3027 - acc: 0.8885 - val_loss: 0.2878 - val_acc: 0.8943
# Epoch 3/100
# 60000/60000 [==============================] - 33s 545us/sample - loss: 0.2552 - acc: 0.9047 - val_loss: 0.2689 - val_acc: 0.9015
# Epoch 4/100
# 60000/60000 [==============================] - 33s 544us/sample - loss: 0.2262 - acc: 0.9168 - val_loss: 0.2488 - val_acc: 0.9109
# Epoch 5/100
# 60000/60000 [==============================] - 33s 547us/sample - loss: 0.2039 - acc: 0.9248 - val_loss: 0.2318 - val_acc: 0.9134
# Epoch 6/100
# 60000/60000 [==============================] - 33s 543us/sample - loss: 0.1841 - acc: 0.9307 - val_loss: 0.2280 - val_acc: 0.9164
# Epoch 7/100
# 60000/60000 [==============================] - 33s 553us/sample - loss: 0.1685 - acc: 0.9380 - val_loss: 0.2466 - val_acc: 0.9106
# Epoch 8/100
# 60000/60000 [==============================] - 34s 559us/sample - loss: 0.1511 - acc: 0.9438 - val_loss: 0.2356 - val_acc: 0.9164
# Epoch 9/100
# 60000/60000 [==============================] - 33s 549us/sample - loss: 0.1385 - acc: 0.9497 - val_loss: 0.2130 - val_acc: 0.9272
# Epoch 10/100
# 60000/60000 [==============================] - 33s 547us/sample - loss: 0.1246 - acc: 0.9534 - val_loss: 0.3292 - val_acc: 0.8863
# Epoch 11/100
# 60000/60000 [==============================] - 32s 539us/sample - loss: 0.1154 - acc: 0.9571 - val_loss: 0.2323 - val_acc: 0.9252
# Epoch 12/100
# 60000/60000 [==============================] - 34s 562us/sample - loss: 0.1021 - acc: 0.9616 - val_loss: 0.2277 - val_acc: 0.9276
# Epoch 13/100
# 60000/60000 [==============================] - 33s 555us/sample - loss: 0.0925 - acc: 0.9653 - val_loss: 0.2350 - val_acc: 0.9267
# Epoch 14/100
# 60000/60000 [==============================] - 33s 556us/sample - loss: 0.0832 - acc: 0.9689 - val_loss: 0.2464 - val_acc: 0.9250
# Epoch 15/100
# 60000/60000 [==============================] - 34s 562us/sample - loss: 0.0751 - acc: 0.9715 - val_loss: 0.2603 - val_acc: 0.9252
# Epoch 16/100
# 60000/60000 [==============================] - 33s 556us/sample - loss: 0.0699 - acc: 0.9730 - val_loss: 0.3485 - val_acc: 0.9098
# Epoch 17/100
# 60000/60000 [==============================] - 33s 554us/sample - loss: 0.0596 - acc: 0.9783 - val_loss: 0.2487 - val_acc: 0.9308
# Epoch 18/100
# 60000/60000 [==============================] - 32s 541us/sample - loss: 0.0548 - acc: 0.9801 - val_loss: 0.2591 - val_acc: 0.9300
# Epoch 19/100
# 60000/60000 [==============================] - 33s 544us/sample - loss: 0.0485 - acc: 0.9817 - val_loss: 0.2560 - val_acc: 0.9324
# Epoch 20/100
# 60000/60000 [==============================] - 33s 542us/sample - loss: 0.0456 - acc: 0.9823 - val_loss: 0.2750 - val_acc: 0.9341
# Epoch 21/100
# 60000/60000 [==============================] - 32s 540us/sample - loss: 0.0406 - acc: 0.9851 - val_loss: 0.2959 - val_acc: 0.9331
# Epoch 22/100
# 60000/60000 [==============================] - 33s 553us/sample - loss: 0.0371 - acc: 0.9862 - val_loss: 0.3008 - val_acc: 0.9287
# Epoch 23/100
# 60000/60000 [==============================] - 33s 552us/sample - loss: 0.0307 - acc: 0.9885 - val_loss: 0.3239 - val_acc: 0.9238
# Epoch 24/100
# 60000/60000 [==============================] - 33s 543us/sample - loss: 0.0291 - acc: 0.9890 - val_loss: 0.3229 - val_acc: 0.9325
# Epoch 25/100
# 60000/60000 [==============================] - 33s 549us/sample - loss: 0.0290 - acc: 0.9894 - val_loss: 0.3187 - val_acc: 0.9337
# Epoch 26/100
# 60000/60000 [==============================] - 33s 545us/sample - loss: 0.0268 - acc: 0.9902 - val_loss: 0.3115 - val_acc: 0.9312
# Epoch 27/100
# 60000/60000 [==============================] - 33s 543us/sample - loss: 0.0241 - acc: 0.9916 - val_loss: 0.3489 - val_acc: 0.9280
# Epoch 28/100
# 60000/60000 [==============================] - 33s 542us/sample - loss: 0.0214 - acc: 0.9926 - val_loss: 0.3268 - val_acc: 0.9345
# Epoch 29/100
# 60000/60000 [==============================] - 33s 553us/sample - loss: 0.0223 - acc: 0.9920 - val_loss: 0.5048 - val_acc: 0.9116
# Epoch 30/100
# 60000/60000 [==============================] - 34s 566us/sample - loss: 0.0188 - acc: 0.9932 - val_loss: 0.3395 - val_acc: 0.9337
# Epoch 31/100
# 60000/60000 [==============================] - 33s 552us/sample - loss: 0.0174 - acc: 0.9937 - val_loss: 0.3717 - val_acc: 0.9334
# Epoch 32/100
# 60000/60000 [==============================] - 33s 548us/sample - loss: 0.0181 - acc: 0.9936 - val_loss: 0.3651 - val_acc: 0.9331
# Epoch 33/100
# 60000/60000 [==============================] - 32s 540us/sample - loss: 0.0159 - acc: 0.9941 - val_loss: 0.3892 - val_acc: 0.9318
# Epoch 34/100
# 60000/60000 [==============================] - 33s 542us/sample - loss: 0.0158 - acc: 0.9947 - val_loss: 0.4070 - val_acc: 0.9295
# Epoch 35/100
# 60000/60000 [==============================] - 33s 545us/sample - loss: 0.0126 - acc: 0.9956 - val_loss: 0.3919 - val_acc: 0.9322
# Epoch 36/100
# 60000/60000 [==============================] - 33s 543us/sample - loss: 0.0126 - acc: 0.9957 - val_loss: 0.4024 - val_acc: 0.9308
# Epoch 37/100
# 60000/60000 [==============================] - 33s 547us/sample - loss: 0.0138 - acc: 0.9951 - val_loss: 0.4049 - val_acc: 0.9352
# Epoch 38/100
# 60000/60000 [==============================] - 33s 543us/sample - loss: 0.0143 - acc: 0.9950 - val_loss: 0.3984 - val_acc: 0.9293
# Epoch 39/100
# 60000/60000 [==============================] - 33s 557us/sample - loss: 0.0121 - acc: 0.9958 - val_loss: 0.3989 - val_acc: 0.9321
# Epoch 40/100
# 60000/60000 [==============================] - 34s 558us/sample - loss: 0.0100 - acc: 0.9963 - val_loss: 0.3869 - val_acc: 0.9331
# Epoch 41/100
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0117 - acc: 0.9956 - val_loss: 0.4137 - val_acc: 0.9317
# Epoch 42/100
# 60000/60000 [==============================] - 32s 541us/sample - loss: 0.0101 - acc: 0.9967 - val_loss: 0.4196 - val_acc: 0.9298
# Epoch 43/100
# 60000/60000 [==============================] - 33s 542us/sample - loss: 0.0103 - acc: 0.9963 - val_loss: 0.4387 - val_acc: 0.9297
# Epoch 44/100
# 60000/60000 [==============================] - 33s 549us/sample - loss: 0.0091 - acc: 0.9969 - val_loss: 0.4183 - val_acc: 0.9324
# Epoch 45/100
# 60000/60000 [==============================] - 33s 557us/sample - loss: 0.0092 - acc: 0.9969 - val_loss: 0.4203 - val_acc: 0.9342
# Epoch 46/100
# 60000/60000 [==============================] - 33s 548us/sample - loss: 0.0068 - acc: 0.9976 - val_loss: 0.4216 - val_acc: 0.9336
# Epoch 47/100
# 60000/60000 [==============================] - 33s 552us/sample - loss: 0.0076 - acc: 0.9973 - val_loss: 0.3944 - val_acc: 0.9340
# Epoch 48/100
# 60000/60000 [==============================] - 33s 543us/sample - loss: 0.0061 - acc: 0.9980 - val_loss: 0.4199 - val_acc: 0.9361
# Epoch 49/100
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0066 - acc: 0.9978 - val_loss: 0.4243 - val_acc: 0.9339
# Epoch 50/100
# 60000/60000 [==============================] - 32s 540us/sample - loss: 0.0060 - acc: 0.9980 - val_loss: 0.4655 - val_acc: 0.9343
# Epoch 51/100
# 60000/60000 [==============================] - 33s 542us/sample - loss: 0.0060 - acc: 0.9981 - val_loss: 0.4398 - val_acc: 0.9337
# Epoch 52/100
# 60000/60000 [==============================] - 33s 547us/sample - loss: 0.0070 - acc: 0.9976 - val_loss: 0.4569 - val_acc: 0.9310
# Epoch 53/100
# 60000/60000 [==============================] - 32s 540us/sample - loss: 0.0058 - acc: 0.9979 - val_loss: 0.4243 - val_acc: 0.9326
# Epoch 54/100
# 60000/60000 [==============================] - 32s 540us/sample - loss: 0.0054 - acc: 0.9982 - val_loss: 0.4462 - val_acc: 0.9331
# Epoch 55/100
# 60000/60000 [==============================] - 32s 540us/sample - loss: 0.0047 - acc: 0.9984 - val_loss: 0.5435 - val_acc: 0.9229
# Epoch 56/100
# 60000/60000 [==============================] - 33s 543us/sample - loss: 0.0050 - acc: 0.9983 - val_loss: 0.4400 - val_acc: 0.9341
# Epoch 57/100
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0051 - acc: 0.9984 - val_loss: 0.4567 - val_acc: 0.9319
# Epoch 58/100
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0041 - acc: 0.9985 - val_loss: 0.4527 - val_acc: 0.9352
# Epoch 59/100
# 60000/60000 [==============================] - 33s 543us/sample - loss: 0.0064 - acc: 0.9976 - val_loss: 0.4506 - val_acc: 0.9334
# Epoch 60/100
# 60000/60000 [==============================] - 33s 552us/sample - loss: 0.0055 - acc: 0.9981 - val_loss: 0.4508 - val_acc: 0.9322
# Epoch 61/100
# 60000/60000 [==============================] - 34s 563us/sample - loss: 0.0048 - acc: 0.9982 - val_loss: 0.4801 - val_acc: 0.9312
# Epoch 62/100
# 60000/60000 [==============================] - 34s 564us/sample - loss: 0.0058 - acc: 0.9980 - val_loss: 0.4538 - val_acc: 0.9326
# Epoch 63/100
# 60000/60000 [==============================] - 34s 563us/sample - loss: 0.0037 - acc: 0.9988 - val_loss: 0.4914 - val_acc: 0.9321
# Epoch 64/100
# 60000/60000 [==============================] - 33s 545us/sample - loss: 0.0061 - acc: 0.9979 - val_loss: 0.4741 - val_acc: 0.9351
# Epoch 65/100
# 60000/60000 [==============================] - 33s 546us/sample - loss: 0.0043 - acc: 0.9987 - val_loss: 0.4519 - val_acc: 0.9387
# Epoch 66/100
# 60000/60000 [==============================] - 33s 543us/sample - loss: 0.0033 - acc: 0.9990 - val_loss: 0.4806 - val_acc: 0.9347
# Epoch 67/100
# 60000/60000 [==============================] - 33s 545us/sample - loss: 0.0043 - acc: 0.9985 - val_loss: 0.5291 - val_acc: 0.9201
# Epoch 68/100
# 60000/60000 [==============================] - 33s 547us/sample - loss: 0.0043 - acc: 0.9986 - val_loss: 0.4559 - val_acc: 0.9357
# Epoch 69/100
# 60000/60000 [==============================] - 33s 542us/sample - loss: 0.0054 - acc: 0.9981 - val_loss: 0.4822 - val_acc: 0.9299
# Epoch 70/100
# 60000/60000 [==============================] - 33s 542us/sample - loss: 0.0046 - acc: 0.9985 - val_loss: 0.4451 - val_acc: 0.9336
# Epoch 71/100
# 60000/60000 [==============================] - 33s 544us/sample - loss: 0.0049 - acc: 0.9984 - val_loss: 0.4449 - val_acc: 0.9331
# Epoch 72/100
# 60000/60000 [==============================] - 32s 541us/sample - loss: 0.0031 - acc: 0.9989 - val_loss: 0.4707 - val_acc: 0.9352
# Epoch 73/100
# 60000/60000 [==============================] - 32s 540us/sample - loss: 0.0026 - acc: 0.9991 - val_loss: 0.4636 - val_acc: 0.9351
# Epoch 74/100
# 60000/60000 [==============================] - 32s 540us/sample - loss: 0.0029 - acc: 0.9990 - val_loss: 0.4716 - val_acc: 0.9325
# Epoch 75/100
# 60000/60000 [==============================] - 32s 541us/sample - loss: 0.0034 - acc: 0.9989 - val_loss: 0.5043 - val_acc: 0.9342
# Epoch 76/100
# 60000/60000 [==============================] - 32s 539us/sample - loss: 0.0039 - acc: 0.9987 - val_loss: 0.4768 - val_acc: 0.9364
# Epoch 77/100
# 60000/60000 [==============================] - 32s 540us/sample - loss: 0.0023 - acc: 0.9993 - val_loss: 0.4985 - val_acc: 0.9319
# Epoch 78/100
# 60000/60000 [==============================] - 32s 541us/sample - loss: 0.0028 - acc: 0.9990 - val_loss: 0.5074 - val_acc: 0.9334
# Epoch 79/100
# 60000/60000 [==============================] - 32s 541us/sample - loss: 0.0039 - acc: 0.9986 - val_loss: 0.5087 - val_acc: 0.9324
# Epoch 80/100
# 60000/60000 [==============================] - 33s 545us/sample - loss: 0.0033 - acc: 0.9989 - val_loss: 0.6077 - val_acc: 0.9222
# Epoch 81/100
# 60000/60000 [==============================] - 33s 545us/sample - loss: 0.0044 - acc: 0.9984 - val_loss: 0.4897 - val_acc: 0.9333
# Epoch 82/100
# 60000/60000 [==============================] - 33s 545us/sample - loss: 0.0029 - acc: 0.9990 - val_loss: 0.4875 - val_acc: 0.9343
# Epoch 83/100
# 60000/60000 [==============================] - 32s 542us/sample - loss: 0.0020 - acc: 0.9993 - val_loss: 0.5162 - val_acc: 0.9334
# Epoch 84/100
# 60000/60000 [==============================] - 32s 541us/sample - loss: 0.0038 - acc: 0.9986 - val_loss: 0.4688 - val_acc: 0.9331
# Epoch 85/100
# 60000/60000 [==============================] - 33s 544us/sample - loss: 0.0024 - acc: 0.9992 - val_loss: 0.4739 - val_acc: 0.9338
# Epoch 86/100
# 60000/60000 [==============================] - 33s 543us/sample - loss: 0.0018 - acc: 0.9994 - val_loss: 0.4888 - val_acc: 0.9318
# Epoch 87/100
# 60000/60000 [==============================] - 33s 544us/sample - loss: 0.0026 - acc: 0.9991 - val_loss: 0.5039 - val_acc: 0.9352
# Epoch 88/100
# 60000/60000 [==============================] - 33s 552us/sample - loss: 0.0022 - acc: 0.9994 - val_loss: 0.4873 - val_acc: 0.9355
# Epoch 89/100
# 60000/60000 [==============================] - 34s 560us/sample - loss: 0.0030 - acc: 0.9989 - val_loss: 0.4834 - val_acc: 0.9342
# Epoch 90/100
# 60000/60000 [==============================] - 33s 543us/sample - loss: 0.0021 - acc: 0.9993 - val_loss: 0.5180 - val_acc: 0.9321
# Epoch 91/100
# 60000/60000 [==============================] - 33s 542us/sample - loss: 0.0020 - acc: 0.9993 - val_loss: 0.5143 - val_acc: 0.9350
# Epoch 92/100
# 60000/60000 [==============================] - 32s 541us/sample - loss: 0.0029 - acc: 0.9991 - val_loss: 0.4726 - val_acc: 0.9358
# Epoch 93/100
# 60000/60000 [==============================] - 32s 540us/sample - loss: 0.0022 - acc: 0.9994 - val_loss: 0.5073 - val_acc: 0.9359
# Epoch 94/100
# 60000/60000 [==============================] - 32s 542us/sample - loss: 0.0019 - acc: 0.9994 - val_loss: 0.5218 - val_acc: 0.9354
# Epoch 95/100
# 60000/60000 [==============================] - 32s 541us/sample - loss: 0.0020 - acc: 0.9993 - val_loss: 0.5105 - val_acc: 0.9371
# Epoch 96/100
# 60000/60000 [==============================] - 32s 541us/sample - loss: 0.0014 - acc: 0.9995 - val_loss: 0.5115 - val_acc: 0.9366
# Epoch 97/100
# 60000/60000 [==============================] - 33s 542us/sample - loss: 0.0017 - acc: 0.9994 - val_loss: 0.5046 - val_acc: 0.9364
# Epoch 98/100
# 60000/60000 [==============================] - 32s 540us/sample - loss: 0.0012 - acc: 0.9997 - val_loss: 0.5331 - val_acc: 0.9313
# Epoch 99/100
# 60000/60000 [==============================] - 32s 542us/sample - loss: 0.0017 - acc: 0.9996 - val_loss: 0.5157 - val_acc: 0.9335
# Epoch 100/100
# 60000/60000 [==============================] - 32s 540us/sample - loss: 0.0022 - acc: 0.9992 - val_loss: 0.5179 - val_acc: 0.9337
# 10000/10000 [==============================] - 2s 200us/sample - loss: 0.5179 - acc: 0.9337








