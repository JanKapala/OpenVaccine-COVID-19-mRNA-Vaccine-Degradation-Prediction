import tensorflow as tf
from spektral.layers import GraphAttention
from spektral.layers import GlobalAttentionPool


class LSTMGraphReduceCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.conv_graph_layer = GraphAttention(units)
        self.pool_graph_layer = GlobalAttentionPool(units)

        self.W_f_a = self.add_weight('W_forget_a', shape=[self.units, self.units])
        self.W_f_x = self.add_weight('W_forget_x', shape=[self.units, self.units])
        self.b_f = self.add_weight('b_forget', shape=[self.units],
                                   initializer='random_normal', trainable=True)

        self.W_u_a = self.add_weight('W_update_a', shape=[self.units, self.units])
        self.W_u_x = self.add_weight('W_update_x', shape=[self.units, self.units])
        self.b_u = self.add_weight('b_update', shape=[self.units],
                                   initializer='random_normal', trainable=True)

        self.W_o_a = self.add_weight('W_output_a', shape=[self.units, self.units])
        self.W_o_x = self.add_weight('W_output_x', shape=[self.units, self.units])
        self.b_o = self.add_weight('b_output', shape=[self.units],
                                   initializer='random_normal', trainable=True)

        self.W_c_a = self.add_weight('W_cell_a', shape=[self.units, self.units])
        self.W_c_x = self.add_weight('W_cell_x', shape=[self.units, self.units])
        self.b_c = self.add_weight('b_cell', shape=[self.units],
                                   initializer='random_normal', trainable=True)

    @property
    def state_size(self):
        return [self.units, self.units]

    @property
    def output_size(self):
        return self.units

    def call(self, inputs, states):
        # Unpack input and state
        features, adj_matrix, edges_features_matrix = inputs
        node_features = self.conv_graph_layer([features, adj_matrix, edges_features_matrix])
        x = self.pool_graph_layer(node_features)
        c, a = states

        # Calculate forget and update gates values
        forget_g = tf.sigmoid(tf.matmul(a, self.W_f_a) + tf.matmul(x, self.W_f_x) + self.b_f)
        update_g = tf.sigmoid(tf.matmul(a, self.W_u_a) + tf.matmul(x, self.W_u_x) + self.b_u)

        # Calculate update value for cell state
        c_dash = tf.tanh(tf.matmul(a, self.W_c_a) + tf.matmul(x, self.W_c_x) + self.b_c)

        # Calculate new cell state as weighted average
        next_c = forget_g * c + update_g * c_dash

        # Calculate output gate values
        output_g = tf.sigmoid(tf.matmul(a, self.W_o_a) + tf.matmul(x, self.W_o_x) + self.b_o)

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
        config.update({'units': self.units})
                       # 'conv_graph_layer': self.conv_graph_layer,
                       # 'pool_graph_layer': self.pool_graph_layer})
        return config

