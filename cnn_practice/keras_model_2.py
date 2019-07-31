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
    layers.Dense(2048, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

# Train on 60000 samples, validate on 10000 samples
# Epoch 1/50
# 60000/60000 [==============================] - 29s 475us/sample - loss: 1.0175 - acc: 0.8372 - val_loss: 0.3768 - val_acc: 0.8650
# Epoch 2/50
# 60000/60000 [==============================] - 27s 451us/sample - loss: 0.2891 - acc: 0.8949 - val_loss: 0.3039 - val_acc: 0.8909
# Epoch 3/50
# 60000/60000 [==============================] - 27s 448us/sample - loss: 0.2404 - acc: 0.9117 - val_loss: 0.3072 - val_acc: 0.8891
# Epoch 4/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.2094 - acc: 0.9233 - val_loss: 0.3032 - val_acc: 0.8878
# Epoch 5/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.1858 - acc: 0.9324 - val_loss: 0.2699 - val_acc: 0.9034
# Epoch 6/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.1679 - acc: 0.9392 - val_loss: 0.2955 - val_acc: 0.8936
# Epoch 7/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.1518 - acc: 0.9457 - val_loss: 0.2634 - val_acc: 0.9073
# Epoch 8/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.1378 - acc: 0.9510 - val_loss: 0.2701 - val_acc: 0.9056
# Epoch 9/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.1250 - acc: 0.9559 - val_loss: 0.2719 - val_acc: 0.9083
# Epoch 10/50
# 60000/60000 [==============================] - 27s 445us/sample - loss: 0.1143 - acc: 0.9605 - val_loss: 0.2562 - val_acc: 0.9107
# Epoch 11/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.1038 - acc: 0.9648 - val_loss: 0.2544 - val_acc: 0.9141
# Epoch 12/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0943 - acc: 0.9685 - val_loss: 0.2608 - val_acc: 0.9106
# Epoch 13/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.0866 - acc: 0.9720 - val_loss: 0.2588 - val_acc: 0.9132
# Epoch 14/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0782 - acc: 0.9754 - val_loss: 0.2671 - val_acc: 0.9121
# Epoch 15/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0719 - acc: 0.9777 - val_loss: 0.2807 - val_acc: 0.9097
# Epoch 16/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0661 - acc: 0.9802 - val_loss: 0.2714 - val_acc: 0.9148
# Epoch 17/50
# 60000/60000 [==============================] - 27s 445us/sample - loss: 0.0594 - acc: 0.9830 - val_loss: 0.2772 - val_acc: 0.9130
# Epoch 18/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.0543 - acc: 0.9844 - val_loss: 0.2707 - val_acc: 0.9164
# Epoch 19/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.0489 - acc: 0.9868 - val_loss: 0.2779 - val_acc: 0.9130
# Epoch 20/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.0453 - acc: 0.9876 - val_loss: 0.2714 - val_acc: 0.9189
# Epoch 21/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0412 - acc: 0.9892 - val_loss: 0.2776 - val_acc: 0.9166
# Epoch 22/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0372 - acc: 0.9906 - val_loss: 0.2949 - val_acc: 0.9140
# Epoch 23/50
# 60000/60000 [==============================] - 27s 449us/sample - loss: 0.0340 - acc: 0.9918 - val_loss: 0.2843 - val_acc: 0.9181
# Epoch 24/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0309 - acc: 0.9933 - val_loss: 0.2931 - val_acc: 0.9149
# Epoch 25/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.0283 - acc: 0.9939 - val_loss: 0.2887 - val_acc: 0.9186
# Epoch 26/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.0259 - acc: 0.9948 - val_loss: 0.2953 - val_acc: 0.9176
# Epoch 27/50
# 60000/60000 [==============================] - 27s 448us/sample - loss: 0.0233 - acc: 0.9960 - val_loss: 0.2985 - val_acc: 0.9191
# Epoch 28/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0212 - acc: 0.9962 - val_loss: 0.3030 - val_acc: 0.9186
# Epoch 29/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.0198 - acc: 0.9965 - val_loss: 0.3011 - val_acc: 0.9193
# Epoch 30/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.0179 - acc: 0.9973 - val_loss: 0.3126 - val_acc: 0.9168
# Epoch 31/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0164 - acc: 0.9977 - val_loss: 0.3103 - val_acc: 0.9193
# Epoch 32/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.0151 - acc: 0.9981 - val_loss: 0.3184 - val_acc: 0.9184
# Epoch 33/50
# 60000/60000 [==============================] - 27s 445us/sample - loss: 0.0138 - acc: 0.9982 - val_loss: 0.3149 - val_acc: 0.9196
# Epoch 34/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.0127 - acc: 0.9985 - val_loss: 0.3246 - val_acc: 0.9187
# Epoch 35/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0115 - acc: 0.9989 - val_loss: 0.3279 - val_acc: 0.9183
# Epoch 36/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.0106 - acc: 0.9991 - val_loss: 0.3286 - val_acc: 0.9186
# Epoch 37/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.0097 - acc: 0.9992 - val_loss: 0.3322 - val_acc: 0.9191
# Epoch 38/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.0091 - acc: 0.9993 - val_loss: 0.3328 - val_acc: 0.9188
# Epoch 39/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0085 - acc: 0.9993 - val_loss: 0.3362 - val_acc: 0.9190
# Epoch 40/50
# 60000/60000 [==============================] - 27s 445us/sample - loss: 0.0078 - acc: 0.9994 - val_loss: 0.3373 - val_acc: 0.9207
# Epoch 41/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.0071 - acc: 0.9997 - val_loss: 0.3422 - val_acc: 0.9194
# Epoch 42/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0067 - acc: 0.9996 - val_loss: 0.3447 - val_acc: 0.9194
# Epoch 43/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0062 - acc: 0.9998 - val_loss: 0.3520 - val_acc: 0.9177
# Epoch 44/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0057 - acc: 0.9998 - val_loss: 0.3522 - val_acc: 0.9197
# Epoch 45/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0054 - acc: 0.9999 - val_loss: 0.3649 - val_acc: 0.9185
# Epoch 46/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.0050 - acc: 0.9999 - val_loss: 0.3564 - val_acc: 0.9186
# Epoch 47/50
# 60000/60000 [==============================] - 27s 447us/sample - loss: 0.0048 - acc: 0.9998 - val_loss: 0.3557 - val_acc: 0.9190
# Epoch 48/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0044 - acc: 1.0000 - val_loss: 0.3586 - val_acc: 0.9196
# Epoch 49/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0043 - acc: 0.9999 - val_loss: 0.3593 - val_acc: 0.9198
# Epoch 50/50
# 60000/60000 [==============================] - 27s 446us/sample - loss: 0.0040 - acc: 0.9999 - val_loss: 0.3614 - val_acc: 0.9202
# 10000/10000 [==============================] - 1s 85us/sample - loss: 0.3614 - acc: 0.9202

