# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = np.reshape(x_train, (60000, 28, 28, 1))
x_test = np.reshape(x_test, (10000, 28, 28, 1))

# x_train, y_train = x_train[:10000], y_train[:10000]
x_val, y_val = x_test[:1000], y_test[:1000]

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
y_val = tf.keras.utils.to_categorical(y_val, 10)

model = tf.keras.models.Sequential([
    layers.Conv2D(64, (2, 2), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPool2D((2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(128, (2, 2), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dense(500, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

# Train on 60000 samples, validate on 10000 samples
# Epoch 1/30
# 60000/60000 [==============================] - 145s 2ms/sample - loss: 0.4226 - acc: 0.8486 - val_loss: 0.3344 - val_acc: 0.8773
# Epoch 2/30
# 60000/60000 [==============================] - 145s 2ms/sample - loss: 0.2888 - acc: 0.8937 - val_loss: 0.2931 - val_acc: 0.8906
# Epoch 3/30
# 60000/60000 [==============================] - 144s 2ms/sample - loss: 0.2480 - acc: 0.9082 - val_loss: 0.2910 - val_acc: 0.8917
# Epoch 4/30
# 60000/60000 [==============================] - 146s 2ms/sample - loss: 0.2202 - acc: 0.9184 - val_loss: 0.2570 - val_acc: 0.9063
# Epoch 5/30
# 60000/60000 [==============================] - 150s 3ms/sample - loss: 0.1975 - acc: 0.9281 - val_loss: 0.2485 - val_acc: 0.9101
# Epoch 6/30
# 60000/60000 [==============================] - 147s 2ms/sample - loss: 0.1797 - acc: 0.9335 - val_loss: 0.2734 - val_acc: 0.9002
# Epoch 7/30
# 60000/60000 [==============================] - 144s 2ms/sample - loss: 0.1630 - acc: 0.9395 - val_loss: 0.2497 - val_acc: 0.9079
# Epoch 8/30
# 60000/60000 [==============================] - 146s 2ms/sample - loss: 0.1466 - acc: 0.9459 - val_loss: 0.2519 - val_acc: 0.9127
# Epoch 9/30
# 60000/60000 [==============================] - 143s 2ms/sample - loss: 0.1340 - acc: 0.9510 - val_loss: 0.2558 - val_acc: 0.9114
# Epoch 10/30
# 60000/60000 [==============================] - 147s 2ms/sample - loss: 0.1209 - acc: 0.9551 - val_loss: 0.2552 - val_acc: 0.9098
# Epoch 11/30
# 60000/60000 [==============================] - 154s 3ms/sample - loss: 0.1074 - acc: 0.9611 - val_loss: 0.2640 - val_acc: 0.9119
# Epoch 12/30
# 60000/60000 [==============================] - 152s 3ms/sample - loss: 0.0957 - acc: 0.9647 - val_loss: 0.2678 - val_acc: 0.9120
# Epoch 13/30
# 60000/60000 [==============================] - 147s 2ms/sample - loss: 0.0860 - acc: 0.9691 - val_loss: 0.2626 - val_acc: 0.9157
# Epoch 14/30
# 60000/60000 [==============================] - 146s 2ms/sample - loss: 0.0765 - acc: 0.9720 - val_loss: 0.2723 - val_acc: 0.9156
# Epoch 15/30
# 60000/60000 [==============================] - 158s 3ms/sample - loss: 0.0671 - acc: 0.9765 - val_loss: 0.2918 - val_acc: 0.9150
# Epoch 16/30
# 60000/60000 [==============================] - 163s 3ms/sample - loss: 0.0591 - acc: 0.9785 - val_loss: 0.2888 - val_acc: 0.9160
# Epoch 17/30
# 60000/60000 [==============================] - 175s 3ms/sample - loss: 0.0524 - acc: 0.9816 - val_loss: 0.2913 - val_acc: 0.9162
# Epoch 18/30
# 60000/60000 [==============================] - 175s 3ms/sample - loss: 0.0454 - acc: 0.9840 - val_loss: 0.3097 - val_acc: 0.9170
# Epoch 19/30
# 60000/60000 [==============================] - 161s 3ms/sample - loss: 0.0398 - acc: 0.9865 - val_loss: 0.3121 - val_acc: 0.9155
# Epoch 20/30
# 60000/60000 [==============================] - 148s 2ms/sample - loss: 0.0328 - acc: 0.9893 - val_loss: 0.3320 - val_acc: 0.9147
# Epoch 21/30
# 60000/60000 [==============================] - 144s 2ms/sample - loss: 0.0271 - acc: 0.9910 - val_loss: 0.3868 - val_acc: 0.9046
# Epoch 22/30
# 60000/60000 [==============================] - 142s 2ms/sample - loss: 0.0231 - acc: 0.9929 - val_loss: 0.3507 - val_acc: 0.9161
# Epoch 23/30
# 60000/60000 [==============================] - 146s 2ms/sample - loss: 0.0190 - acc: 0.9946 - val_loss: 0.3583 - val_acc: 0.9185
# Epoch 24/30
# 60000/60000 [==============================] - 143s 2ms/sample - loss: 0.0181 - acc: 0.9947 - val_loss: 0.3808 - val_acc: 0.9147
# Epoch 25/30
# 60000/60000 [==============================] - 143s 2ms/sample - loss: 0.0177 - acc: 0.9947 - val_loss: 0.3805 - val_acc: 0.9173
# Epoch 26/30
# 60000/60000 [==============================] - 145s 2ms/sample - loss: 0.0118 - acc: 0.9972 - val_loss: 0.3829 - val_acc: 0.9149
# Epoch 27/30
# 60000/60000 [==============================] - 143s 2ms/sample - loss: 0.0083 - acc: 0.9982 - val_loss: 0.4146 - val_acc: 0.9147
# Epoch 28/30
# 60000/60000 [==============================] - 143s 2ms/sample - loss: 0.0086 - acc: 0.9978 - val_loss: 0.3911 - val_acc: 0.9166
# Epoch 29/30
# 60000/60000 [==============================] - 185s 3ms/sample - loss: 0.0103 - acc: 0.9981 - val_loss: 0.4778 - val_acc: 0.8980
# Epoch 30/30
# 60000/60000 [==============================] - 163s 3ms/sample - loss: 0.0075 - acc: 0.9985 - val_loss: 0.4038 - val_acc: 0.9202
# 10000/10000 [==============================] - 7s 705us/sample - loss: 0.4038 - acc: 0.9202
