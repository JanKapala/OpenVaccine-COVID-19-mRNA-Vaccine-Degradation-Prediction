import tensorflow as tf
import numpy as np
import os


class SequenceGenerator(tf.keras.layers.Layer):
    def __init__(self, rnn, seq_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Prepare internal RNN
        self.rnn = rnn
        self.rnn.return_sequences = True

        # Define output shape by specifying init_tensor_shape
        self.vec_len = self.rnn.units
        self.init_tensor_shape = [seq_len, 1]

        # Initialize random tensor for feeding internal RNN
        self.init_tensor = tf.random.uniform(self.init_tensor_shape)

    def call(self, initial_state, training=None, mask=None):
        # Dynamically get a batch size
        batch_size = tf.shape(initial_state[0])[0]

        # Dynamically Tile init_tensor to match batch dimension
        init_tensor_batch = tf.reshape(tf.tile(self.init_tensor, [batch_size, 1]),
                                       [batch_size] + self.init_tensor_shape)

        # Call internal rnn
        output = self.rnn(init_tensor_batch, initial_state=initial_state,
                          training=training, mask=mask)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({'rnn': self.rnn,
                       'vec_len': self.vec_len,
                       'init_tensor_shape': self.init_tensor_shape,
                       'init_tensor': self.init_tensor.numpy()})
        return config

    @classmethod
    def from_config(cls, config):
        config['init_tensor'] = tf.constant(config['init_tensor'])
        return cls(**config)

if __name__ == '__main__':
    # TEST

    # Create symbolic initial state input tensors
    initial_state_h_input = tf.keras.Input(shape=(5,))
    initial_state_c_input = tf.keras.Input(shape=(5,))
    initial_state_input = [initial_state_h_input, initial_state_c_input]
    # print('symbolic initial_state')
    # print(initial_state_input)

    # Create sequence generator
    sequence_generator = SequenceGenerator(tf.keras.layers.LSTM(5), seq_len=3)

    # Get symbolic output tensor
    x = sequence_generator(initial_state_input)
    # print(x)

    # Create model from layer for saving purpose
    sequence_generator_model = tf.keras.Model(inputs=initial_state_input, outputs=x)

    # Save it
    MODEL_NAME = 'sequence_generator_model_tmp'
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    MODEL_PATH = os.path.join(SCRIPT_DIR, MODEL_NAME)
    sequence_generator_model.save(MODEL_PATH)

    # Load it back
    sequence_generator_model_loaded = tf.keras.models.load_model(MODEL_PATH)

    # Create eager initial state input tensors
    initial_state_h = tf.constant(np.random.rand(2, 5), dtype=tf.float32)
    initial_state_c = tf.constant(np.random.rand(2, 5), dtype=tf.float32)
    initial_state = [initial_state_h, initial_state_c]
    # print('initial_state')
    # print(initial_state)

    # Get outputs from layer, model and loaded model
    o1 = sequence_generator(initial_state)
    o2 = sequence_generator_model(initial_state)
    o3 = sequence_generator_model_loaded(initial_state)

    # Check if all works in the same way
    assert np.all(np.isclose(o1, o3))
    assert np.all(np.isclose(o1, o2))
