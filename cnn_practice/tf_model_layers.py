import tensorflow as tf
import math


def sigmoid(z):
    with tf.name_scope('sigmoid'):
        return 1. / (1. + tf.exp(-z))


def relu(z):
    return tf.maximum(0., z, name='relu')


def softmax(logits):
    with tf.name_scope('softmax'):
        logits_exp = tf.exp(logits)
        logits_sum = tf.reduce_sum(logits_exp, 1)
        a = tf.matmul(tf.linalg.tensor_diag(1 / logits_sum), logits_exp)
    return a


def cross_entropy(y_hat, y):
    with tf.name_scope('cross_entropy'):
        return tf.negative(tf.add(y * tf.math.log(y_hat), (1 - y) * tf.math.log(1 - y_hat)))


class Flatten(object):
    def __call__(self, *args, **kwargs):
        with tf.name_scope('flatten'):
            x = args[0]
            a = tf.reshape(x, [x.shape[0].value, -1])
        return a


class Dense(object):
    def __init__(self, units, activation=None):
        self.units = units
        self.activation = activation

    def __call__(self, *args, **kwargs):
        with tf.name_scope('dense'):
            x = args[0]
            w_shape = [x.shape[1].value, self.units]
            b_shape = [self.units]

            self.w = tf.Variable(tf.random.normal(w_shape, stddev=0.1),
                                 name='weights', shape=w_shape)
            self.b = tf.Variable(tf.zeros(b_shape),
                                 name='biases', shape=b_shape)

            z = tf.linalg.matmul(x, self.w) + self.b

            if self.activation:
                a = self.activation(z)
            else:
                a = z

        return a


class Conv2D(object):
    def __init__(self, filters, kernel_size):
        self.filters = filters
        self.kernel_size = kernel_size

    def __call__(self, *args, **kwargs):
        with tf.name_scope('conv_2d'):
            x = args[0]

            w_shape = [self.kernel_size ** 2 * x.shape[3].value, self.filters]
            b_shape = [self.filters]

            self.w = tf.Variable(tf.random.normal(w_shape, stddev=0.1),
                                 name='weights', shape=w_shape)
            self.b = tf.Variable(tf.zeros(b_shape),
                                 name='biases', shape=b_shape)

            patched_x = tf.image.extract_image_patches(
                x,
                [1, self.kernel_size, self.kernel_size, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                padding='SAME'
            )

            z = relu(tf.linalg.matvec(self.w, patched_x, transpose_a=True) + self.b)
            return z


class MaxPool2D(object):
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def __call__(self, *args, **kwargs):
        with tf.name_scope('max_pool_2d'):
            x = args[0]
            return tf.nn.max_pool2d(x, self.pool_size, self.pool_size, padding='VALID')


class BatchNormalization(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        x = args[0]
        return tf.keras.layers.BatchNormalization()(x)


class Dropout(object):
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, *args, **kwargs):
        x = args[0]
        return tf.nn.dropout(x, rate=self.rate)
