import tensorflow as tf


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial,name="Real_Fake_Classification_Weight",trainable=True)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name="Real_Fake_Classification_Bias",trainable=True)