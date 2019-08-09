import tensorflow as tf


def relu(z):
    return tf.maximum(0., z, name='relu')


def softmax(logits):
    with tf.name_scope('softmax'):
        logits_exp = tf.exp(logits)
        logits_sum = tf.reduce_sum(logits_exp)
        a = logits_exp / logits_sum
    return a


class Flatten(object):
    def __call__(self, *args, **kwargs):
        with tf.name_scope('flatten'):
            x = args[0]
            a = tf.reshape(x, [1, tf.reduce_prod(x.shape)])
        return a


class Dense(object):
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

    def __call__(self, *args, **kwargs):
        with tf.name_scope('dense'):
            x = args[0]
            w_shape = [x.shape[1], self.units]
            b_shape = [self.units]

            self.w = tf.Variable(tf.ones(w_shape), name='weights', shape=w_shape, dtype=tf.float32)
            self.b = tf.Variable(tf.ones(b_shape), name='biases', shape=b_shape, dtype=tf.float32)

            z = tf.linalg.matmul(x, self.w) + self.b
            a = self.activation(z)
        return a


class Conv2D(object):
    def __init__(self, filters, kernel_size):
        self.filters = filters
        self.kernel_size = kernel_size

    def __call__(self, *args, **kwargs):
        pass


class MaxPool2D(object):
    def __init__(self, pool_size):
        self.pool_size = pool_size


class BatchNormalization(object):
    def __init__(self):
        pass


class Dropout(object):
    def __init__(self, rate):
        self.rate = rate
