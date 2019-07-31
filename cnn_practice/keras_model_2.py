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
    layers.Dense(2048, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

# Train on 60000 samples, validate on 10000 samples
# Epoch 1/50
# 60000/60000 [==============================] - 29s 478us/sample - loss: 0.8221 - acc: 0.8237 - val_loss: 0.3714 - val_acc: 0.8638
# Epoch 2/50
# 60000/60000 [==============================] - 28s 462us/sample - loss: 0.3385 - acc: 0.8775 - val_loss: 0.3260 - val_acc: 0.8821
# Epoch 3/50
# 60000/60000 [==============================] - 27s 453us/sample - loss: 0.2982 - acc: 0.8914 - val_loss: 0.2931 - val_acc: 0.8922
# Epoch 4/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.2741 - acc: 0.8985 - val_loss: 0.2860 - val_acc: 0.8951
# Epoch 5/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.2553 - acc: 0.9054 - val_loss: 0.2782 - val_acc: 0.8982
# Epoch 6/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.2369 - acc: 0.9119 - val_loss: 0.2694 - val_acc: 0.9033
# Epoch 7/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.2240 - acc: 0.9175 - val_loss: 0.2596 - val_acc: 0.9069
# Epoch 8/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.2160 - acc: 0.9201 - val_loss: 0.2555 - val_acc: 0.9088
# Epoch 9/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.2074 - acc: 0.9233 - val_loss: 0.2663 - val_acc: 0.9034
# Epoch 10/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.1982 - acc: 0.9260 - val_loss: 0.2578 - val_acc: 0.9073
# Epoch 11/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.1914 - acc: 0.9283 - val_loss: 0.2465 - val_acc: 0.9134
# Epoch 12/50
# 60000/60000 [==============================] - 27s 453us/sample - loss: 0.1812 - acc: 0.9324 - val_loss: 0.2483 - val_acc: 0.9116
# Epoch 13/50
# 60000/60000 [==============================] - 28s 461us/sample - loss: 0.1764 - acc: 0.9347 - val_loss: 0.2480 - val_acc: 0.9118
# Epoch 14/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.1686 - acc: 0.9379 - val_loss: 0.2449 - val_acc: 0.9155
# Epoch 15/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.1649 - acc: 0.9388 - val_loss: 0.2472 - val_acc: 0.9111
# Epoch 16/50
# 60000/60000 [==============================] - 27s 453us/sample - loss: 0.1587 - acc: 0.9413 - val_loss: 0.2422 - val_acc: 0.9178
# Epoch 17/50
# 60000/60000 [==============================] - 27s 451us/sample - loss: 0.1524 - acc: 0.9437 - val_loss: 0.2408 - val_acc: 0.9167
# Epoch 18/50
# 60000/60000 [==============================] - 28s 465us/sample - loss: 0.1506 - acc: 0.9443 - val_loss: 0.2479 - val_acc: 0.9120
# Epoch 19/50
# 60000/60000 [==============================] - 28s 463us/sample - loss: 0.1454 - acc: 0.9466 - val_loss: 0.2407 - val_acc: 0.9141
# Epoch 20/50
# 60000/60000 [==============================] - 27s 453us/sample - loss: 0.1426 - acc: 0.9476 - val_loss: 0.2454 - val_acc: 0.9135
# Epoch 21/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.1381 - acc: 0.9490 - val_loss: 0.2420 - val_acc: 0.9160
# Epoch 22/50
# 60000/60000 [==============================] - 27s 457us/sample - loss: 0.1337 - acc: 0.9497 - val_loss: 0.2437 - val_acc: 0.9171
# Epoch 23/50
# 60000/60000 [==============================] - 28s 461us/sample - loss: 0.1310 - acc: 0.9520 - val_loss: 0.2455 - val_acc: 0.9161
# Epoch 24/50
# 60000/60000 [==============================] - 27s 451us/sample - loss: 0.1282 - acc: 0.9524 - val_loss: 0.2471 - val_acc: 0.9154
# Epoch 25/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.1222 - acc: 0.9558 - val_loss: 0.2529 - val_acc: 0.9122
# Epoch 26/50
# 60000/60000 [==============================] - 27s 453us/sample - loss: 0.1203 - acc: 0.9557 - val_loss: 0.2533 - val_acc: 0.9136
# Epoch 27/50
# 60000/60000 [==============================] - 27s 453us/sample - loss: 0.1180 - acc: 0.9568 - val_loss: 0.2461 - val_acc: 0.9162
# Epoch 28/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.1138 - acc: 0.9592 - val_loss: 0.2457 - val_acc: 0.9173
# Epoch 29/50
# 60000/60000 [==============================] - 27s 453us/sample - loss: 0.1110 - acc: 0.9598 - val_loss: 0.2463 - val_acc: 0.9163
# Epoch 30/50
# 60000/60000 [==============================] - 27s 451us/sample - loss: 0.1092 - acc: 0.9597 - val_loss: 0.2488 - val_acc: 0.9170
# Epoch 31/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.1052 - acc: 0.9613 - val_loss: 0.2453 - val_acc: 0.9168
# Epoch 32/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.1036 - acc: 0.9628 - val_loss: 0.2494 - val_acc: 0.9167
# Epoch 33/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.1005 - acc: 0.9636 - val_loss: 0.2492 - val_acc: 0.9176
# Epoch 34/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.1014 - acc: 0.9627 - val_loss: 0.2500 - val_acc: 0.9175
# Epoch 35/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.0955 - acc: 0.9655 - val_loss: 0.2493 - val_acc: 0.9172
# Epoch 36/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.0944 - acc: 0.9660 - val_loss: 0.2491 - val_acc: 0.9184
# Epoch 37/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.0937 - acc: 0.9654 - val_loss: 0.2553 - val_acc: 0.9173
# Epoch 38/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.0916 - acc: 0.9666 - val_loss: 0.2500 - val_acc: 0.9196
# Epoch 39/50
# 60000/60000 [==============================] - 27s 451us/sample - loss: 0.0881 - acc: 0.9681 - val_loss: 0.2534 - val_acc: 0.9169
# Epoch 40/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.0860 - acc: 0.9683 - val_loss: 0.2558 - val_acc: 0.9162
# Epoch 41/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.0846 - acc: 0.9691 - val_loss: 0.2580 - val_acc: 0.9164
# Epoch 42/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.0829 - acc: 0.9699 - val_loss: 0.2506 - val_acc: 0.9205
# Epoch 43/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.0803 - acc: 0.9712 - val_loss: 0.2547 - val_acc: 0.9188
# Epoch 44/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.0813 - acc: 0.9700 - val_loss: 0.2568 - val_acc: 0.9189
# Epoch 45/50
# 60000/60000 [==============================] - 27s 451us/sample - loss: 0.0773 - acc: 0.9723 - val_loss: 0.2575 - val_acc: 0.9185
# Epoch 46/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.0755 - acc: 0.9736 - val_loss: 0.2628 - val_acc: 0.9186
# Epoch 47/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.0738 - acc: 0.9729 - val_loss: 0.2618 - val_acc: 0.9191
# Epoch 48/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.0747 - acc: 0.9728 - val_loss: 0.2569 - val_acc: 0.9209
# Epoch 49/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.0727 - acc: 0.9738 - val_loss: 0.2620 - val_acc: 0.9191
# Epoch 50/50
# 60000/60000 [==============================] - 27s 452us/sample - loss: 0.0706 - acc: 0.9742 - val_loss: 0.2539 - val_acc: 0.9224
# 10000/10000 [==============================] - 1s 85us/sample - loss: 0.2539 - acc: 0.9224
#

