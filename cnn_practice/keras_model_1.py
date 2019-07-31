# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = np.reshape(x_train, (60000, 28, 28, 1))
x_test = np.reshape(x_test, (10000, 28, 28, 1))

# x_train, y_train = x_train[:10000], y_train[:10000]
# x_test, y_test = x_test[:1000], y_test[:1000]

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.models.Sequential([
    layers.Conv2D(64, (2, 2), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPool2D((2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(128, (2, 2), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

# Train on 60000 samples, validate on 10000 samples
# Epoch 1/50
# 60000/60000 [==============================] - 15s 258us/sample - loss: 0.4118 - acc: 0.8549 - val_loss: 0.3343 - val_acc: 0.8782
# Epoch 2/50
# 60000/60000 [==============================] - 14s 229us/sample - loss: 0.2779 - acc: 0.8981 - val_loss: 0.3461 - val_acc: 0.8742
# Epoch 3/50
# 60000/60000 [==============================] - 14s 229us/sample - loss: 0.2268 - acc: 0.9165 - val_loss: 0.2760 - val_acc: 0.9035
# Epoch 4/50
# 60000/60000 [==============================] - 14s 229us/sample - loss: 0.1977 - acc: 0.9277 - val_loss: 0.2716 - val_acc: 0.9027
# Epoch 5/50
# 60000/60000 [==============================] - 14s 229us/sample - loss: 0.1727 - acc: 0.9378 - val_loss: 0.2948 - val_acc: 0.8985
# Epoch 6/50
# 60000/60000 [==============================] - 14s 228us/sample - loss: 0.1545 - acc: 0.9436 - val_loss: 0.2628 - val_acc: 0.9093
# Epoch 7/50
# 60000/60000 [==============================] - 14s 227us/sample - loss: 0.1347 - acc: 0.9504 - val_loss: 0.2642 - val_acc: 0.9089
# Epoch 8/50
# 60000/60000 [==============================] - 14s 231us/sample - loss: 0.1179 - acc: 0.9568 - val_loss: 0.2692 - val_acc: 0.9074
# Epoch 9/50
# 60000/60000 [==============================] - 14s 231us/sample - loss: 0.1031 - acc: 0.9629 - val_loss: 0.2863 - val_acc: 0.9059
# Epoch 10/50
# 60000/60000 [==============================] - 14s 229us/sample - loss: 0.0927 - acc: 0.9667 - val_loss: 0.2990 - val_acc: 0.9064
# Epoch 11/50
# 60000/60000 [==============================] - 14s 229us/sample - loss: 0.0803 - acc: 0.9711 - val_loss: 0.2816 - val_acc: 0.9096
# Epoch 12/50
# 60000/60000 [==============================] - 14s 230us/sample - loss: 0.0719 - acc: 0.9748 - val_loss: 0.2953 - val_acc: 0.9125
# Epoch 13/50
# 60000/60000 [==============================] - 14s 229us/sample - loss: 0.0625 - acc: 0.9779 - val_loss: 0.3146 - val_acc: 0.9087
# Epoch 14/50
# 60000/60000 [==============================] - 14s 230us/sample - loss: 0.0550 - acc: 0.9810 - val_loss: 0.3142 - val_acc: 0.9091
# Epoch 15/50
# 60000/60000 [==============================] - 14s 230us/sample - loss: 0.0497 - acc: 0.9827 - val_loss: 0.3119 - val_acc: 0.9160
# Epoch 16/50
# 60000/60000 [==============================] - 14s 230us/sample - loss: 0.0447 - acc: 0.9839 - val_loss: 0.3186 - val_acc: 0.9153
# Epoch 17/50
# 60000/60000 [==============================] - 14s 226us/sample - loss: 0.0396 - acc: 0.9861 - val_loss: 0.3273 - val_acc: 0.9146
# Epoch 18/50
# 60000/60000 [==============================] - 14s 229us/sample - loss: 0.0354 - acc: 0.9881 - val_loss: 0.3214 - val_acc: 0.9159
# Epoch 19/50
# 60000/60000 [==============================] - 14s 232us/sample - loss: 0.0308 - acc: 0.9897 - val_loss: 0.3407 - val_acc: 0.9124
# Epoch 20/50
# 60000/60000 [==============================] - 14s 233us/sample - loss: 0.0287 - acc: 0.9908 - val_loss: 0.3376 - val_acc: 0.9149
# Epoch 21/50
# 60000/60000 [==============================] - 14s 229us/sample - loss: 0.0266 - acc: 0.9913 - val_loss: 0.3497 - val_acc: 0.9129
# Epoch 22/50
# 60000/60000 [==============================] - 14s 231us/sample - loss: 0.0246 - acc: 0.9920 - val_loss: 0.3414 - val_acc: 0.9152
# Epoch 23/50
# 60000/60000 [==============================] - 14s 230us/sample - loss: 0.0199 - acc: 0.9941 - val_loss: 0.3656 - val_acc: 0.9105
# Epoch 24/50
# 60000/60000 [==============================] - 14s 231us/sample - loss: 0.0188 - acc: 0.9942 - val_loss: 0.3606 - val_acc: 0.9155
# Epoch 25/50
# 60000/60000 [==============================] - 14s 228us/sample - loss: 0.0183 - acc: 0.9940 - val_loss: 0.3741 - val_acc: 0.9115
# Epoch 26/50
# 60000/60000 [==============================] - 14s 229us/sample - loss: 0.0163 - acc: 0.9950 - val_loss: 0.3843 - val_acc: 0.9125
# Epoch 27/50
# 60000/60000 [==============================] - 14s 230us/sample - loss: 0.0155 - acc: 0.9952 - val_loss: 0.3620 - val_acc: 0.9201
# Epoch 28/50
# 60000/60000 [==============================] - 14s 230us/sample - loss: 0.0143 - acc: 0.9959 - val_loss: 0.3730 - val_acc: 0.9176
# Epoch 29/50
# 60000/60000 [==============================] - 14s 229us/sample - loss: 0.0140 - acc: 0.9961 - val_loss: 0.3853 - val_acc: 0.9160
# Epoch 30/50
# 60000/60000 [==============================] - 14s 228us/sample - loss: 0.0110 - acc: 0.9969 - val_loss: 0.4669 - val_acc: 0.9041
# Epoch 31/50
# 60000/60000 [==============================] - 14s 228us/sample - loss: 0.0100 - acc: 0.9975 - val_loss: 0.4197 - val_acc: 0.9124
# Epoch 32/50
# 60000/60000 [==============================] - 14s 230us/sample - loss: 0.0110 - acc: 0.9967 - val_loss: 0.3781 - val_acc: 0.9172
# Epoch 33/50
# 60000/60000 [==============================] - 14s 228us/sample - loss: 0.0090 - acc: 0.9979 - val_loss: 0.3879 - val_acc: 0.9177
# Epoch 34/50
# 60000/60000 [==============================] - 14s 228us/sample - loss: 0.0084 - acc: 0.9979 - val_loss: 0.3918 - val_acc: 0.9151
# Epoch 35/50
# 60000/60000 [==============================] - 14s 230us/sample - loss: 0.0080 - acc: 0.9979 - val_loss: 0.4194 - val_acc: 0.9147
# Epoch 36/50
# 60000/60000 [==============================] - 14s 229us/sample - loss: 0.0078 - acc: 0.9982 - val_loss: 0.3879 - val_acc: 0.9195
# Epoch 37/50
# 60000/60000 [==============================] - 14s 231us/sample - loss: 0.0074 - acc: 0.9981 - val_loss: 0.4251 - val_acc: 0.9122
# Epoch 38/50
# 60000/60000 [==============================] - 14s 228us/sample - loss: 0.0074 - acc: 0.9980 - val_loss: 0.3996 - val_acc: 0.9174
# Epoch 39/50
# 60000/60000 [==============================] - 14s 229us/sample - loss: 0.0065 - acc: 0.9985 - val_loss: 0.4059 - val_acc: 0.9178
# Epoch 40/50
# 60000/60000 [==============================] - 14s 231us/sample - loss: 0.0068 - acc: 0.9984 - val_loss: 0.4109 - val_acc: 0.9155
# Epoch 41/50
# 60000/60000 [==============================] - 14s 230us/sample - loss: 0.0066 - acc: 0.9983 - val_loss: 0.3994 - val_acc: 0.9186
# Epoch 42/50
# 60000/60000 [==============================] - 14s 231us/sample - loss: 0.0064 - acc: 0.9982 - val_loss: 0.4058 - val_acc: 0.9161
# Epoch 43/50
# 60000/60000 [==============================] - 14s 230us/sample - loss: 0.0067 - acc: 0.9984 - val_loss: 0.4244 - val_acc: 0.9163
# Epoch 44/50
# 60000/60000 [==============================] - 14s 230us/sample - loss: 0.0064 - acc: 0.9983 - val_loss: 0.4090 - val_acc: 0.9169
# Epoch 45/50
# 60000/60000 [==============================] - 14s 230us/sample - loss: 0.0060 - acc: 0.9985 - val_loss: 0.4130 - val_acc: 0.9184
# Epoch 46/50
# 60000/60000 [==============================] - 14s 230us/sample - loss: 0.0061 - acc: 0.9985 - val_loss: 0.4135 - val_acc: 0.9170
# Epoch 47/50
# 60000/60000 [==============================] - 14s 231us/sample - loss: 0.0056 - acc: 0.9985 - val_loss: 0.4187 - val_acc: 0.9180
# Epoch 48/50
# 60000/60000 [==============================] - 14s 233us/sample - loss: 0.0042 - acc: 0.9991 - val_loss: 0.4189 - val_acc: 0.9183
# Epoch 49/50
# 60000/60000 [==============================] - 14s 232us/sample - loss: 0.0041 - acc: 0.9991 - val_loss: 0.4115 - val_acc: 0.9197
# Epoch 50/50
# 60000/60000 [==============================] - 14s 229us/sample - loss: 0.0037 - acc: 0.9992 - val_loss: 0.4186 - val_acc: 0.9205
# 10000/10000 [==============================] - 1s 82us/sample - loss: 0.4186 - acc: 0.9205

