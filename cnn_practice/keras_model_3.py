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
    layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.SGD(0.05)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)

# Train on 60000 samples, validate on 10000 samples
# Epoch 1/100
# 60000/60000 [==============================] - 40s 662us/sample - loss: 0.4342 - acc: 0.8548 - val_loss: 0.3822 - val_acc: 0.8499
# Epoch 2/100
# 60000/60000 [==============================] - 37s 619us/sample - loss: 0.2554 - acc: 0.9064 - val_loss: 0.2708 - val_acc: 0.9001
# Epoch 3/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 0.2005 - acc: 0.9269 - val_loss: 0.2512 - val_acc: 0.9111
# Epoch 4/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 0.1692 - acc: 0.9364 - val_loss: 0.2411 - val_acc: 0.9157
# Epoch 5/100
# 60000/60000 [==============================] - 37s 616us/sample - loss: 0.1400 - acc: 0.9467 - val_loss: 0.2303 - val_acc: 0.9232
# Epoch 6/100
# 60000/60000 [==============================] - 37s 615us/sample - loss: 0.1165 - acc: 0.9561 - val_loss: 0.2654 - val_acc: 0.9180
# Epoch 7/100
# 60000/60000 [==============================] - 37s 619us/sample - loss: 0.0950 - acc: 0.9644 - val_loss: 0.2532 - val_acc: 0.9207
# Epoch 8/100
# 60000/60000 [==============================] - 37s 616us/sample - loss: 0.0743 - acc: 0.9720 - val_loss: 0.2986 - val_acc: 0.9200
# Epoch 9/100
# 60000/60000 [==============================] - 37s 616us/sample - loss: 0.0632 - acc: 0.9764 - val_loss: 0.3361 - val_acc: 0.9147
# Epoch 10/100
# 60000/60000 [==============================] - 37s 616us/sample - loss: 0.0515 - acc: 0.9804 - val_loss: 0.3194 - val_acc: 0.9248
# Epoch 11/100
# 60000/60000 [==============================] - 37s 615us/sample - loss: 0.0411 - acc: 0.9848 - val_loss: 0.2885 - val_acc: 0.9321
# Epoch 12/100
# 60000/60000 [==============================] - 37s 616us/sample - loss: 0.0326 - acc: 0.9886 - val_loss: 0.3233 - val_acc: 0.9323
# Epoch 13/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 0.0273 - acc: 0.9902 - val_loss: 0.3330 - val_acc: 0.9301
# Epoch 14/100
# 60000/60000 [==============================] - 37s 619us/sample - loss: 0.0280 - acc: 0.9899 - val_loss: 0.3707 - val_acc: 0.9230
# Epoch 15/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 0.0252 - acc: 0.9909 - val_loss: 0.3389 - val_acc: 0.9297
# Epoch 16/100
# 60000/60000 [==============================] - 37s 616us/sample - loss: 0.0209 - acc: 0.9925 - val_loss: 0.3849 - val_acc: 0.9285
# Epoch 17/100
# 60000/60000 [==============================] - 37s 616us/sample - loss: 0.0160 - acc: 0.9944 - val_loss: 0.3799 - val_acc: 0.9274
# Epoch 18/100
# 60000/60000 [==============================] - 37s 616us/sample - loss: 0.0124 - acc: 0.9958 - val_loss: 0.3733 - val_acc: 0.9299
# Epoch 19/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 0.0110 - acc: 0.9963 - val_loss: 0.3651 - val_acc: 0.9338
# Epoch 20/100
# 60000/60000 [==============================] - 37s 616us/sample - loss: 0.0080 - acc: 0.9974 - val_loss: 0.3890 - val_acc: 0.9323
# Epoch 21/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 0.0070 - acc: 0.9978 - val_loss: 0.4054 - val_acc: 0.9309
# Epoch 22/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 0.0082 - acc: 0.9970 - val_loss: 0.4078 - val_acc: 0.9284
# Epoch 23/100
# 60000/60000 [==============================] - 37s 623us/sample - loss: 0.0041 - acc: 0.9987 - val_loss: 0.4160 - val_acc: 0.9337
# Epoch 24/100
# 60000/60000 [==============================] - 37s 620us/sample - loss: 0.0079 - acc: 0.9973 - val_loss: 0.4660 - val_acc: 0.9221
# Epoch 25/100
# 60000/60000 [==============================] - 37s 616us/sample - loss: 0.0073 - acc: 0.9977 - val_loss: 0.4170 - val_acc: 0.9354
# Epoch 26/100
# 60000/60000 [==============================] - 37s 620us/sample - loss: 0.0081 - acc: 0.9973 - val_loss: 0.4127 - val_acc: 0.9355
# Epoch 27/100
# 60000/60000 [==============================] - 37s 619us/sample - loss: 0.0074 - acc: 0.9977 - val_loss: 0.4296 - val_acc: 0.9329
# Epoch 28/100
# 60000/60000 [==============================] - 37s 616us/sample - loss: 0.0073 - acc: 0.9978 - val_loss: 0.4238 - val_acc: 0.9333
# Epoch 29/100
# 60000/60000 [==============================] - 37s 616us/sample - loss: 0.0031 - acc: 0.9991 - val_loss: 0.4293 - val_acc: 0.9345
# Epoch 30/100
# 60000/60000 [==============================] - 38s 626us/sample - loss: 0.0029 - acc: 0.9991 - val_loss: 0.4378 - val_acc: 0.9353
# Epoch 31/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 0.0023 - acc: 0.9994 - val_loss: 0.5588 - val_acc: 0.9215
# Epoch 32/100
# 60000/60000 [==============================] - 37s 619us/sample - loss: 0.0049 - acc: 0.9984 - val_loss: 0.4463 - val_acc: 0.9343
# Epoch 33/100
# 60000/60000 [==============================] - 38s 626us/sample - loss: 0.0021 - acc: 0.9995 - val_loss: 0.4432 - val_acc: 0.9345
# Epoch 34/100
# 60000/60000 [==============================] - 71s 1ms/sample - loss: 0.0012 - acc: 0.9996 - val_loss: 0.4419 - val_acc: 0.9394
# Epoch 35/100
# 60000/60000 [==============================] - 37s 610us/sample - loss: 0.0015 - acc: 0.9996 - val_loss: 0.4338 - val_acc: 0.9354
# Epoch 36/100
# 60000/60000 [==============================] - 36s 601us/sample - loss: 0.0026 - acc: 0.9993 - val_loss: 0.4452 - val_acc: 0.9343
# Epoch 37/100
# 60000/60000 [==============================] - 36s 599us/sample - loss: 0.0013 - acc: 0.9996 - val_loss: 0.4662 - val_acc: 0.9337
# Epoch 38/100
# 60000/60000 [==============================] - 37s 615us/sample - loss: 0.0014 - acc: 0.9996 - val_loss: 0.4604 - val_acc: 0.9359
# Epoch 39/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 0.0014 - acc: 0.9995 - val_loss: 0.4456 - val_acc: 0.9365
# Epoch 40/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 2.8019e-04 - acc: 1.0000 - val_loss: 0.4515 - val_acc: 0.9363
# Epoch 41/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 3.6375e-04 - acc: 0.9999 - val_loss: 0.4616 - val_acc: 0.9353
# Epoch 42/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 4.0166e-04 - acc: 0.9999 - val_loss: 0.4760 - val_acc: 0.9339
# Epoch 43/100
# 60000/60000 [==============================] - 37s 619us/sample - loss: 2.0669e-04 - acc: 1.0000 - val_loss: 0.4670 - val_acc: 0.9372
# Epoch 44/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 1.1954e-04 - acc: 1.0000 - val_loss: 0.4804 - val_acc: 0.9358
# Epoch 45/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 1.1946e-04 - acc: 1.0000 - val_loss: 0.4811 - val_acc: 0.9358
# Epoch 46/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 5.9509e-05 - acc: 1.0000 - val_loss: 0.4792 - val_acc: 0.9364
# Epoch 47/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 6.9591e-05 - acc: 1.0000 - val_loss: 0.4828 - val_acc: 0.9374
# Epoch 48/100
# 60000/60000 [==============================] - 37s 623us/sample - loss: 9.5711e-05 - acc: 1.0000 - val_loss: 0.4780 - val_acc: 0.9381
# Epoch 49/100
# 60000/60000 [==============================] - 37s 619us/sample - loss: 4.2895e-05 - acc: 1.0000 - val_loss: 0.4840 - val_acc: 0.9371
# Epoch 50/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 4.0459e-05 - acc: 1.0000 - val_loss: 0.4904 - val_acc: 0.9370
# Epoch 51/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 4.7301e-05 - acc: 1.0000 - val_loss: 0.4870 - val_acc: 0.9379
# Epoch 52/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 3.2659e-05 - acc: 1.0000 - val_loss: 0.4858 - val_acc: 0.9382
# Epoch 53/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 2.7568e-05 - acc: 1.0000 - val_loss: 0.4838 - val_acc: 0.9382
# Epoch 54/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 3.7653e-05 - acc: 1.0000 - val_loss: 0.4884 - val_acc: 0.9373
# Epoch 55/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 2.6744e-05 - acc: 1.0000 - val_loss: 0.4870 - val_acc: 0.9389
# Epoch 56/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 3.8627e-05 - acc: 1.0000 - val_loss: 0.4968 - val_acc: 0.9378
# Epoch 57/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 7.3002e-04 - acc: 0.9998 - val_loss: 0.4860 - val_acc: 0.9370
# Epoch 58/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 0.0024 - acc: 0.9992 - val_loss: 0.4816 - val_acc: 0.9339
# Epoch 59/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 3.0542e-04 - acc: 0.9999 - val_loss: 0.5360 - val_acc: 0.9322
# Epoch 60/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 0.0014 - acc: 0.9994 - val_loss: 0.5782 - val_acc: 0.9217
# Epoch 61/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 0.0025 - acc: 0.9992 - val_loss: 0.4819 - val_acc: 0.9341
# Epoch 62/100
# 60000/60000 [==============================] - 37s 622us/sample - loss: 0.0023 - acc: 0.9992 - val_loss: 0.4691 - val_acc: 0.9353
# Epoch 63/100
# 60000/60000 [==============================] - 37s 615us/sample - loss: 8.6022e-04 - acc: 0.9997 - val_loss: 0.4687 - val_acc: 0.9363
# Epoch 64/100
# 60000/60000 [==============================] - 37s 616us/sample - loss: 2.9306e-04 - acc: 0.9999 - val_loss: 0.4676 - val_acc: 0.9387
# Epoch 65/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 5.3938e-04 - acc: 0.9999 - val_loss: 0.4844 - val_acc: 0.9366
# Epoch 66/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 7.4351e-04 - acc: 0.9998 - val_loss: 0.4881 - val_acc: 0.9354
# Epoch 67/100
# 60000/60000 [==============================] - 37s 616us/sample - loss: 4.4562e-04 - acc: 0.9999 - val_loss: 0.4746 - val_acc: 0.9380
# Epoch 68/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 5.4970e-04 - acc: 0.9999 - val_loss: 0.4785 - val_acc: 0.9371
# Epoch 69/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 3.4493e-04 - acc: 0.9999 - val_loss: 0.4844 - val_acc: 0.9356
# Epoch 70/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 1.5375e-04 - acc: 1.0000 - val_loss: 0.4830 - val_acc: 0.9370
# Epoch 71/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 7.9180e-05 - acc: 1.0000 - val_loss: 0.4873 - val_acc: 0.9373
# Epoch 72/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 5.3815e-05 - acc: 1.0000 - val_loss: 0.4927 - val_acc: 0.9376
# Epoch 73/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 3.7830e-05 - acc: 1.0000 - val_loss: 0.4932 - val_acc: 0.9375
# Epoch 74/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 3.5021e-05 - acc: 1.0000 - val_loss: 0.4936 - val_acc: 0.9383
# Epoch 75/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 3.8701e-04 - acc: 0.9999 - val_loss: 0.4925 - val_acc: 0.9357
# Epoch 76/100
# 60000/60000 [==============================] - 37s 616us/sample - loss: 0.0012 - acc: 0.9997 - val_loss: 0.4999 - val_acc: 0.9359
# Epoch 77/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 4.7419e-04 - acc: 0.9998 - val_loss: 0.4850 - val_acc: 0.9377
# Epoch 78/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 1.9198e-04 - acc: 1.0000 - val_loss: 0.4837 - val_acc: 0.9376
# Epoch 79/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 8.9546e-05 - acc: 1.0000 - val_loss: 0.4949 - val_acc: 0.9378
# Epoch 80/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 5.9326e-05 - acc: 1.0000 - val_loss: 0.4993 - val_acc: 0.9386
# Epoch 81/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 8.3572e-05 - acc: 1.0000 - val_loss: 0.5035 - val_acc: 0.9388
# Epoch 82/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 4.1014e-05 - acc: 1.0000 - val_loss: 0.5005 - val_acc: 0.9390
# Epoch 83/100
# 60000/60000 [==============================] - 37s 621us/sample - loss: 4.0268e-05 - acc: 1.0000 - val_loss: 0.5000 - val_acc: 0.9390
# Epoch 84/100
# 60000/60000 [==============================] - 37s 620us/sample - loss: 4.7945e-05 - acc: 1.0000 - val_loss: 0.4990 - val_acc: 0.9381
# Epoch 85/100
# 60000/60000 [==============================] - 37s 623us/sample - loss: 3.2368e-05 - acc: 1.0000 - val_loss: 0.4997 - val_acc: 0.9389
# Epoch 86/100
# 60000/60000 [==============================] - 37s 621us/sample - loss: 2.2272e-05 - acc: 1.0000 - val_loss: 0.5011 - val_acc: 0.9389
# Epoch 87/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 2.4414e-05 - acc: 1.0000 - val_loss: 0.5009 - val_acc: 0.9387
# Epoch 88/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 2.7824e-05 - acc: 1.0000 - val_loss: 0.5013 - val_acc: 0.9391
# Epoch 89/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 2.5107e-05 - acc: 1.0000 - val_loss: 0.5032 - val_acc: 0.9391
# Epoch 90/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 3.3070e-05 - acc: 1.0000 - val_loss: 0.5077 - val_acc: 0.9390
# Epoch 91/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 6.7865e-05 - acc: 1.0000 - val_loss: 0.5106 - val_acc: 0.9387
# Epoch 92/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 2.0350e-05 - acc: 1.0000 - val_loss: 0.5107 - val_acc: 0.9388
# Epoch 93/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 2.7660e-05 - acc: 1.0000 - val_loss: 0.5101 - val_acc: 0.9395
# Epoch 94/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 1.6318e-05 - acc: 1.0000 - val_loss: 0.5088 - val_acc: 0.9400
# Epoch 95/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 1.9830e-05 - acc: 1.0000 - val_loss: 0.5105 - val_acc: 0.9393
# Epoch 96/100
# 60000/60000 [==============================] - 37s 619us/sample - loss: 1.4398e-05 - acc: 1.0000 - val_loss: 0.5121 - val_acc: 0.9394
# Epoch 97/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 4.1492e-05 - acc: 1.0000 - val_loss: 0.5190 - val_acc: 0.9379
# Epoch 98/100
# 60000/60000 [==============================] - 37s 617us/sample - loss: 2.4078e-05 - acc: 1.0000 - val_loss: 0.5137 - val_acc: 0.9393
# Epoch 99/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 1.4880e-05 - acc: 1.0000 - val_loss: 0.5144 - val_acc: 0.9394
# Epoch 100/100
# 60000/60000 [==============================] - 37s 618us/sample - loss: 1.4675e-05 - acc: 1.0000 - val_loss: 0.5117 - val_acc: 0.9397
# 10000/10000 [==============================] - 2s 214us/sample - loss: 0.5117 - acc: 0.9397










