import tensorflow as tf
import numpy as np


class WindowingLayer(tf.keras.layers.Layer):
    def __init__(self, window_size):
        super(WindowingLayer, self).__init__()
        self.window_size = window_size

    def call(self, input):
        batch_size, seq_len, feature_size = input.shape

        first_part = tf.stack([input[:, i: i + self.window_size] for i in range(0, seq_len - self.window_size + 1)],
                              axis=1)
        incomplete_part = [input[:, i: i + self.window_size] for i in range(seq_len - self.window_size + 1, seq_len)]

        second_part = np.zeros([batch_size, self.window_size - 1, self.window_size, feature_size])

        for i in range(self.window_size - 1):
            for j in range(self.window_size - i - 1):
                for k in range(batch_size):
                    second_part[k, i, j] = incomplete_part[i][k][j]

        return tf.concat([first_part, second_part], axis=1)
