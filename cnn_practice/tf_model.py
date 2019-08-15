import tensorflow.compat.v1 as tf
from tensorflow.python import debug as tf_debug

import numpy as np
import math
import tf_model_layers as layers

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255., x_test / 255.

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 超参数配置
epochs = 30
batch_size = 20
train_batch_num = math.ceil(60000 / batch_size)
test_batch_num = math.ceil(10000 / batch_size)


# 创建模型
model = [
    layers.Flatten(),
    layers.Dense(100, activation=layers.relu),
    layers.Dense(10, activation=layers.softmax)
]

x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 28, 28])
y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 10])

a = x
for layer in model:
    a = layer(a)
y_pred = a

# 训练
loss = tf.reduce_mean(layers.cross_entropy(y_pred, y), name='loss_mean')

optimizer = tf.train.AdagradOptimizer(0.01)
train = optimizer.minimize(loss)

correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.Session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.run(tf.global_variables_initializer())

for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch + 1, epochs))
    acc_log = []
    loss_log = []
    for i in range(train_batch_num):
        s, e = i * batch_size, (i + 1) * batch_size
        x_batch, y_batch = x_train[s: e], y_train[s: e]
        pred, loss_val, acc_val, _ = sess.run(
            [y_pred, loss, accuracy, train], {x: x_batch, y: y_batch})
        acc_log.append(acc_val)
        loss_log.append(loss_val)

        if i+1 == train_batch_num:
            end = ' - '
        else:
            end = '\r'

        print('{:5d}/60000 - loss: {:.4f} - acc: {:.4f}'
              .format((i + 1) * batch_size, np.mean(loss_log), np.mean(acc_log)),
              end=end, flush=True)

    acc_log = []
    loss_log = []
    for i in range(test_batch_num):
        s, e = i * batch_size, (i + 1) * batch_size
        x_batch, y_batch = x_test[s: e], y_test[s: e]
        loss_val, acc_val = sess.run([loss, accuracy], {x: x_batch, y: y_batch})
        acc_log.append(acc_val)
        loss_log.append(loss_val)

    print('val_loss: {:.4f} - val_acc: {:.4f}'.format(np.mean(loss_log), np.mean(acc_log)))


# writer = tf.summary.FileWriter('d:\\tf_events')
# writer.add_graph(tf.get_default_graph())
# writer.flush()

