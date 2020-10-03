import tensorflow as tf
from spektral.layers import GraphAttention
from spektral.layers import GlobalAttentionPool

from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations


class GraphLSTMCell(tf.keras.layers.AbstractRNNCell):
    class _GraphEmbedder(tf.keras.layers.Layer):
        def __init__(self, units, name):
            super().__init__(name=name)
            self.conv_graph_layer = GraphAttention(units)
            self.pool_graph_layer = GlobalAttentionPool(units)

        def call(self, inputs, training=None, mask=None):
            features, adj_matrix, edges_features_matrix = inputs
            node_features = self.conv_graph_layer([features, adj_matrix, edges_features_matrix])
            embedded_graph = self.pool_graph_layer(node_features)

            return embedded_graph

    def __init__(self, units,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Gates weights
        self.f_kernel = self.add_weight('forget_kernel', shape=[self.units, self.units],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer)
        self.f_embedder = self._GraphEmbedder(self.units, name='forget_embedder')
        self.f_bias = self.add_weight('forget_bias', shape=[self.units], initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      trainable=True)

        self.u_kernel = self.add_weight('update_kernel', shape=[self.units, self.units],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer)
        self.u_embedder = self._GraphEmbedder(self.units, name='update_embedder')
        self.u_bias = self.add_weight('update_bias', shape=[self.units], initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      trainable=True)

        self.o_kernel = self.add_weight('output_kernel', shape=[self.units, self.units],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer)
        self.o_embedder = self._GraphEmbedder(self.units, name='output_embedder')
        self.o_bias = self.add_weight('output_bias', shape=[self.units], initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      trainable=True)

        # Cell state weights
        self.c_kernel = self.add_weight('cell_kernel', shape=[self.units, self.units],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer)
        self.c_embedder = self._GraphEmbedder(self.units, name='cell_embedder')
        self.c_bias = self.add_weight('cell_bias', shape=[self.units], initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      trainable=True)

    @property
    def state_size(self):
        return [self.units, self.units]

    @property
    def output_size(self):
        return self.units

    def call(self, inputs, states):
        # Unpack input and state
        x = inputs
        c, a = states

        # Calculate forget and update gates values
        forget_g = tf.sigmoid(tf.matmul(a, self.f_kernel) + self.f_embedder(x) + self.f_bias)
        update_g = tf.sigmoid(tf.matmul(a, self.u_kernel) + self.u_embedder(x) + self.u_bias)

        # Calculate update value for cell state
        c_dash = tf.tanh(tf.matmul(a, self.c_kernel) + self.c_embedder(x) + self.c_bias)

        # Calculate new cell state as weighted average
        next_c = forget_g * c + update_g * c_dash

        # Calculate output gate values
        output_g = tf.sigmoid(tf.matmul(a, self.o_kernel) + self.f_embedder(x) + self.o_bias)

        # calculate hidden state
        next_a = output_g * tf.tanh(c)

        # Define output and pack next state
        output = next_a
        next_state = [next_c, next_a]

        return output, next_state

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        state_size_a, state_size_c = self.state_size

        state_size_a = (batch_size, state_size_a)
        state_size_c = (batch_size, state_size_c)

        state_a = tf.zeros(state_size_a, dtype=dtype)
        state_c = tf.zeros(state_size_c, dtype=dtype)

        return [state_a, state_c]
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units,
                       'kernel_initializer': initializers.serialize(self.kernel_initializer),
                       'bias_initializer': initializers.serialize(self.bias_initializer),
                       'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                       'bias_regularizer': regularizers.serialize(self.bias_regularizer)})
        return config


class GraphLSTM(tf.keras.layers.RNN):
    def __init__(self, units, kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        cell = GraphLSTMCell(units, kernel_initializer, bias_initializer)
        kwargs['cell'] = cell
        super().__init__(**kwargs)

