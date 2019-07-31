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
    layers.Conv2D(190, (2, 2), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

# Train on 60000 samples, validate on 10000 samples
# Epoch 1/50
# 60000/60000 [==============================] - 19s 316us/sample - loss: 0.8455 - acc: 0.8145 - val_loss: 0.3579 - val_acc: 0.8713
# Epoch 2/50
# 60000/60000 [==============================] - 17s 290us/sample - loss: 0.3678 - acc: 0.8686 - val_loss: 0.3294 - val_acc: 0.8801
# Epoch 3/50
# 60000/60000 [==============================] - 17s 286us/sample - loss: 0.3206 - acc: 0.8842 - val_loss: 0.3073 - val_acc: 0.8864
# Epoch 4/50
# 60000/60000 [==============================] - 17s 286us/sample - loss: 0.2929 - acc: 0.8925 - val_loss: 0.2902 - val_acc: 0.8943
# Epoch 5/50
# 60000/60000 [==============================] - 17s 289us/sample - loss: 0.2757 - acc: 0.8996 - val_loss: 0.2823 - val_acc: 0.8965
# Epoch 6/50
# 60000/60000 [==============================] - 17s 285us/sample - loss: 0.2594 - acc: 0.9048 - val_loss: 0.2766 - val_acc: 0.8987
# Epoch 7/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.2464 - acc: 0.9097 - val_loss: 0.2697 - val_acc: 0.8999
# Epoch 8/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.2372 - acc: 0.9133 - val_loss: 0.2677 - val_acc: 0.9021
# Epoch 9/50
# 60000/60000 [==============================] - 17s 286us/sample - loss: 0.2270 - acc: 0.9169 - val_loss: 0.2608 - val_acc: 0.9052
# Epoch 10/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.2187 - acc: 0.9191 - val_loss: 0.2616 - val_acc: 0.9051
# Epoch 11/50
# 60000/60000 [==============================] - 17s 283us/sample - loss: 0.2128 - acc: 0.9207 - val_loss: 0.2584 - val_acc: 0.9060
# Epoch 12/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.2000 - acc: 0.9274 - val_loss: 0.2550 - val_acc: 0.9071
# Epoch 13/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.1989 - acc: 0.9264 - val_loss: 0.2524 - val_acc: 0.9097
# Epoch 14/50
# 60000/60000 [==============================] - 17s 289us/sample - loss: 0.1933 - acc: 0.9292 - val_loss: 0.2497 - val_acc: 0.9103
# Epoch 15/50
# 60000/60000 [==============================] - 17s 290us/sample - loss: 0.1857 - acc: 0.9307 - val_loss: 0.2485 - val_acc: 0.9098
# Epoch 16/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.1820 - acc: 0.9319 - val_loss: 0.2487 - val_acc: 0.9111
# Epoch 17/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.1771 - acc: 0.9345 - val_loss: 0.2455 - val_acc: 0.9130
# Epoch 18/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.1731 - acc: 0.9366 - val_loss: 0.2450 - val_acc: 0.9123
# Epoch 19/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.1685 - acc: 0.9375 - val_loss: 0.2427 - val_acc: 0.9138
# Epoch 20/50
# 60000/60000 [==============================] - 17s 283us/sample - loss: 0.1648 - acc: 0.9388 - val_loss: 0.2424 - val_acc: 0.9138
# Epoch 21/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.1621 - acc: 0.9399 - val_loss: 0.2421 - val_acc: 0.9139
# Epoch 22/50
# 60000/60000 [==============================] - 17s 283us/sample - loss: 0.1584 - acc: 0.9423 - val_loss: 0.2416 - val_acc: 0.9144
# Epoch 23/50
# 60000/60000 [==============================] - 17s 283us/sample - loss: 0.1515 - acc: 0.9453 - val_loss: 0.2402 - val_acc: 0.9164
# Epoch 24/50
# 60000/60000 [==============================] - 17s 283us/sample - loss: 0.1511 - acc: 0.9440 - val_loss: 0.2402 - val_acc: 0.9147
# Epoch 25/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.1467 - acc: 0.9467 - val_loss: 0.2397 - val_acc: 0.9143
# Epoch 26/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.1456 - acc: 0.9466 - val_loss: 0.2377 - val_acc: 0.9165
# Epoch 27/50
# 60000/60000 [==============================] - 17s 285us/sample - loss: 0.1410 - acc: 0.9493 - val_loss: 0.2396 - val_acc: 0.9150
# Epoch 28/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.1396 - acc: 0.9482 - val_loss: 0.2414 - val_acc: 0.9158
# Epoch 29/50
# 60000/60000 [==============================] - 17s 290us/sample - loss: 0.1356 - acc: 0.9502 - val_loss: 0.2396 - val_acc: 0.9145
# Epoch 30/50
# 60000/60000 [==============================] - 17s 287us/sample - loss: 0.1344 - acc: 0.9511 - val_loss: 0.2386 - val_acc: 0.9165
# Epoch 31/50
# 60000/60000 [==============================] - 17s 287us/sample - loss: 0.1317 - acc: 0.9509 - val_loss: 0.2381 - val_acc: 0.9175
# Epoch 32/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.1299 - acc: 0.9516 - val_loss: 0.2418 - val_acc: 0.9160
# Epoch 33/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.1275 - acc: 0.9526 - val_loss: 0.2370 - val_acc: 0.9174
# Epoch 34/50
# 60000/60000 [==============================] - 17s 290us/sample - loss: 0.1261 - acc: 0.9538 - val_loss: 0.2383 - val_acc: 0.9159
# Epoch 35/50
# 60000/60000 [==============================] - 17s 288us/sample - loss: 0.1223 - acc: 0.9550 - val_loss: 0.2396 - val_acc: 0.9166
# Epoch 36/50
# 60000/60000 [==============================] - 17s 286us/sample - loss: 0.1226 - acc: 0.9555 - val_loss: 0.2405 - val_acc: 0.9174
# Epoch 37/50
# 60000/60000 [==============================] - 17s 285us/sample - loss: 0.1208 - acc: 0.9560 - val_loss: 0.2405 - val_acc: 0.9175
# Epoch 38/50
# 60000/60000 [==============================] - 17s 285us/sample - loss: 0.1177 - acc: 0.9573 - val_loss: 0.2416 - val_acc: 0.9179
# Epoch 39/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.1170 - acc: 0.9577 - val_loss: 0.2425 - val_acc: 0.9169
# Epoch 40/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.1144 - acc: 0.9584 - val_loss: 0.2395 - val_acc: 0.9185
# Epoch 41/50
# 60000/60000 [==============================] - 17s 283us/sample - loss: 0.1111 - acc: 0.9600 - val_loss: 0.2386 - val_acc: 0.9177
# Epoch 42/50
# 60000/60000 [==============================] - 17s 285us/sample - loss: 0.1115 - acc: 0.9593 - val_loss: 0.2397 - val_acc: 0.9181
# Epoch 43/50
# 60000/60000 [==============================] - 17s 288us/sample - loss: 0.1090 - acc: 0.9610 - val_loss: 0.2401 - val_acc: 0.9181
# Epoch 44/50
# 60000/60000 [==============================] - 17s 285us/sample - loss: 0.1068 - acc: 0.9615 - val_loss: 0.2384 - val_acc: 0.9186
# Epoch 45/50
# 60000/60000 [==============================] - 17s 289us/sample - loss: 0.1078 - acc: 0.9611 - val_loss: 0.2409 - val_acc: 0.9186
# Epoch 46/50
# 60000/60000 [==============================] - 17s 287us/sample - loss: 0.1045 - acc: 0.9617 - val_loss: 0.2404 - val_acc: 0.9178
# Epoch 47/50
# 60000/60000 [==============================] - 17s 284us/sample - loss: 0.1025 - acc: 0.9633 - val_loss: 0.2449 - val_acc: 0.9179
# Epoch 48/50
# 60000/60000 [==============================] - 17s 283us/sample - loss: 0.1021 - acc: 0.9635 - val_loss: 0.2411 - val_acc: 0.9204
# Epoch 49/50
# 60000/60000 [==============================] - 17s 285us/sample - loss: 0.1010 - acc: 0.9633 - val_loss: 0.2413 - val_acc: 0.9189
# Epoch 50/50
# 60000/60000 [==============================] - 17s 285us/sample - loss: 0.0986 - acc: 0.9642 - val_loss: 0.2422 - val_acc: 0.9188
# 10000/10000 [==============================] - 1s 78us/sample - loss: 0.2422 - acc: 0.9188
