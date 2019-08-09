import tensorflow.compat.v1 as tf
from tensorflow.python import debug as tf_debug

import numpy as np
import tf_model_layers as layers

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255. / 10., x_test / 255. / 10.

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

layers = [
    layers.Flatten(),
    layers.Dense(10, activation=layers.softmax)
]

x = tf.placeholder(dtype=tf.float32, shape=(28, 28))
a = x
for layer in layers:
    a = layer(a)
y = a

sess = tf.Session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.run(tf.global_variables_initializer())

for i in range(100):
    print(sess.run(y, {x: x_train[i]}))

# loss = tf.losses.mean_squared_error(labels=y_train, predictions=y_pred)
# print(sess.run(loss))

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
writer.flush()

