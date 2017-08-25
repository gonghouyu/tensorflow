import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


class LSTM(object):
    def __init__(self, config):

        self.config = config
        assert self.config.num_hidden % len(self.config.periods) == 0
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.inputs  = tf.placeholder(tf.float32, shape=[None, self.config.num_steps, self.config.num_input], name="inputs")
        self.targets = tf.placeholder(tf.float32, shape=[None, self.config.num_output], name="targets")

        self._build_model()

        self._init_optimizer()

    def _build_model(self):
        weights = {
            'out': tf.Variable(tf.random_normal([self.config.num_hidden, self.config.num_output]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.config.num_output]))
        }

        self.inputs = tf.transpose(self.inputs, [1, 0, 2])
        self.inputs = tf.reshape(self.inputs, [-1, self.config.num_input])
        self.inputs = tf.split(0, self.config.num_steps, self.inputs)
        lstm_cell = rnn_cell.BasicLSTMCell(self.config.num_hidden, forget_bias=1.0)

        outputs, states = rnn.rnn(lstm_cell, self.inputs, dtype=tf.float32)
        self.prediction = tf.matmul(outputs[-1], weights['out']) + biases['out']

    def _init_optimizer(self):

        # Learning rate decay, note that is self.learning_rate_decay == 1.0,
        # the decay schedule is disabled, i.e. learning rate is constant.
        self.learning_rate = tf.train.exponential_decay(
            self.config.learning_rate,
            self.global_step,
            self.config.learning_rate_step,
            self.config.learning_rate_decay,
            staircase=True
        )
        self.learning_rate = tf.maximum(self.learning_rate, self.config.learning_rate_min)
        tf.scalar_summary("learning_rate", self.learning_rate)

        # Definition of the optimizer and computing gradients operation
        if self.config.optimizer == 'adam':
            # Adam optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.config.optimizer == 'rmsprop':
            # RMSProper optimizer
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif self.config.optimizer == 'adagrad':
            # AdaGrad optimizer
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        else:
            raise ValueError("Unknown optimizer specified")

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(self.prediction,1), tf.argmax(self.targets,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



