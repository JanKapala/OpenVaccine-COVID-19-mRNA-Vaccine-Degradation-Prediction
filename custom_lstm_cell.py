import tensorflow as tf


class CustomLSTMCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units, **kwargs):
        self.units = units
        super().__init__(**kwargs)

    @property
    def state_size(self):
        return [self.units, self.units]

    @property
    def output_size(self):
        return self.units

    def build(self, input_shapes):
        if isinstance(input_shapes, list):
            input_shape = input_shapes[0]
        else:
            input_shape = input_shapes
        feature_size = int(input_shape[-1])

        self.W_f_a = self.add_weight('W_forget_a', shape=[self.units, self.units])
        self.W_f_x = self.add_weight('W_forget_x', shape=[feature_size, self.units])
        self.b_f = self.add_weight('b_forget', shape=[self.units],
                                   initializer='random_normal', trainable=True)

        self.W_u_a = self.add_weight('W_update_a', shape=[self.units, self.units])
        self.W_u_x = self.add_weight('W_update_x', shape=[feature_size, self.units])
        self.b_u = self.add_weight('b_update', shape=[self.units],
                                   initializer='random_normal', trainable=True)

        self.W_o_a = self.add_weight('W_output_a', shape=[self.units, self.units])
        self.W_o_x = self.add_weight('W_output_x', shape=[feature_size, self.units])
        self.b_o = self.add_weight('b_output', shape=[self.units],
                                   initializer='random_normal', trainable=True)

        self.W_c_a = self.add_weight('W_cell_a', shape=[self.units, self.units])
        self.W_c_x = self.add_weight('W_cell_x', shape=[feature_size, self.units])
        self.b_c = self.add_weight('b_cell', shape=[self.units],
                                   initializer='random_normal', trainable=True)

    def call(self, inputs, states):
        # Unpack input and state
        x = inputs
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


if __name__ == '__main__':
    batch_size = 2
    seq_len = 5
    feature_size = 3

    inputs_eager = tf.random.uniform((batch_size, seq_len, feature_size))
    inputs_symbolic = tf.keras.Input((None, feature_size))

    # print(f"inputs_eager: {inputs_eager}")
    # print(f"inputs_symbolic: {inputs_symbolic}")

    units = 20
    cell = CustomLSTMCell(units)
    custom_lstm_layer = tf.keras.layers.RNN(cell)

    outputs_eager = custom_lstm_layer(inputs_eager)
    outputs_symbolic = custom_lstm_layer(inputs_symbolic)

    assert outputs_eager.shape.as_list() == [batch_size, units]
    assert outputs_symbolic.shape.as_list() == [None, units]
