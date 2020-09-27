import tensorflow as tf
from spektral.layers import GraphAttention
from spektral.layers import GlobalAttentionPool


class GraphReduceCell(tf.keras.layers.AbstractRNNCell):
    """Cell class for GraphReduce layer

        This recurrent cell transform graph with `nodes_number` nodes
        and with features vector (of size `feature_size`) for each node
        into its summary vector of size: `units`.

        Arguments:
            units: Positive integer, dimensionality of the output space.

        Call arguments:
            inputs: A tuple of two 3D tensors, with shapes
                `[(batch, nodes_number, feature_size), (batch, nodes_number, nodes_number)]`.
                The first tensor contains features of graph's nodes and the second is its
                adjacency matrix.
            states: A 2D tensor, with shape of `[batch, units]`. For timestep 0,
                the initial state provided by user will be feed to cell. States aren't
                used in any way for now.
    """

    def __init__(self, units, **kwargs):
        self.units = units
        self.conv_graph_layer = GraphAttention(units)
        self.pool_graph_layer = GlobalAttentionPool(units)
        super().__init__(**kwargs)

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def call(self, inputs, states):
        features, adj_matrix, edges_features_matrix = inputs

        node_features = self.conv_graph_layer([features, adj_matrix, edges_features_matrix])
        single_node_channels = self.pool_graph_layer(node_features)

        return single_node_channels, states

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros((batch_size, self.state_size), dtype=dtype)]


if __name__ == '__main__':
    # Test setup
    batch_size = 2
    seq_len = 7
    window_size = 5
    feature_size = 3
    edges_features_size = 3

    # print('DATA DIMESIONS:')
    # print(f"    -> batch_size: {batch_size}")
    # print(f"    -> seq_len: {seq_len}")
    # print(f"    -> window_size: {window_size}")
    # print(f"    -> feature_size: {feature_size}")
    # print(f"    -> edges_features_size: {edges_features_size}")
    # print()

    # print('EAGER TENSORS')
    features_batch = tf.random.uniform((batch_size, seq_len, window_size, feature_size))
    adj_matrices_batch = tf.random.uniform((batch_size, seq_len, window_size, window_size))
    edges_features_matrices_batch = tf.random.uniform((batch_size, seq_len, window_size, window_size,
                                                       edges_features_size))
    inputs_batch = (features_batch, adj_matrices_batch, edges_features_matrices_batch)
    # print('Batch of full sequences:')
    # print(f"    -> features_batch.shape: {features_batch.shape}")
    # print(f"    -> adj_matrices_batch.shape: {adj_matrices_batch.shape}")
    # print(f"    -> edges_features_matrices_batch.shape: {edges_features_matrices_batch.shape}")
    # print()

    features_batch_at_t = features_batch[:, 0, :, :]
    adj_matrices_batch_at_t = adj_matrices_batch[:, 0, :, :]
    edges_features_matrices_batch_at_t = edges_features_matrices_batch[:, 0, :, :, :]
    inputs_batch_at_t = (features_batch_at_t, adj_matrices_batch_at_t, edges_features_matrices_batch_at_t)
    # print('Batch of one timestep:')
    # print(f"    -> features_batch_at_t.shape: {features_batch_at_t.shape}")
    # print(f"    -> adj_matrices_batch_at_t.shape: {adj_matrices_batch_at_t.shape}")
    # print(f"    -> edges_features_matrices_batch_at_t.shape: {edges_features_matrices_batch_at_t.shape}")
    # print()

    # print('SYMBOLIC TENSORS')
    features_batch_symbolic = tf.keras.layers.Input((None, window_size, feature_size))
    adj_matrices_batch_symbolic = tf.keras.layers.Input((None, window_size, window_size))
    edges_features_matrices_batch_symbolic = tf.keras.layers.Input((None, window_size, window_size,
                                                                    edges_features_size))
    inputs_batch_symbolic = (features_batch_symbolic, adj_matrices_batch_symbolic,
                             edges_features_matrices_batch_symbolic)
    # print('Batch of full sequences:')
    # print(f"    -> features_batch_symbolic.shape: {features_batch_symbolic.shape}")
    # print(f"    -> adj_matrices_batch_symbolic.shape: {adj_matrices_batch_symbolic.shape}")
    # print(f"    -> edges_features_matrices_batch_symbolic.shape: {edges_features_matrices_batch_symbolic.shape}")
    # print()

    features_batch_symbolic_at_t = features_batch_symbolic[:, 0, :, :]
    adj_matrices_batch_symbolic_at_t = adj_matrices_batch_symbolic[:, 0, :, :]
    edges_features_matrices_batch_symbolic_at_t = edges_features_matrices_batch_symbolic[:, 0, :, :, :]
    inputs_batch_symbolic_at_t = (features_batch_symbolic_at_t, adj_matrices_batch_symbolic_at_t,
                                  edges_features_matrices_batch_symbolic_at_t)
    # print('Batch of one timestep:')
    # print(f"    -> features_batch_symbolic_at_t.shape: {features_batch_symbolic_at_t.shape}")
    # print(f"    -> adj_matrices_batch_symbolic_at_t.shape: {adj_matrices_batch_symbolic_at_t.shape}")
    # print(f"    -> edges_features_matrices_batch_symbolic_at_t.shape: "
    #       f"{edges_features_matrices_batch_symbolic_at_t.shape}")

    # GraphReduceCell test
    units = 11
    cell = GraphReduceCell(units)

    cell_output_eager, cell_state_eager = cell(inputs_batch_at_t, cell.get_initial_state(inputs_batch_at_t, batch_size,
                                                                                         inputs_batch_at_t[0].dtype))
    cell_output_symbolic, cell_state_symbolic = cell(inputs_batch_symbolic_at_t,
                                                     cell.get_initial_state(inputs_batch_symbolic_at_t, batch_size,
                                                                            inputs_batch_at_t[0].dtype))

    assert cell_output_eager.shape.as_list() == [batch_size, units]
    assert cell_output_symbolic.shape.as_list() == [None, units]

    # RNN with GraphReduceCell test
    rnn = tf.keras.layers.RNN(cell)

    rnn_output_eager = rnn(inputs_batch)
    rnn_output_symbolic = rnn(inputs_batch_symbolic)

    assert rnn_output_eager.shape.as_list() == [batch_size, units]
    assert rnn_output_symbolic.shape.as_list() == [None, units]

    # RNN with GraphReduceCell test
    rnn_seq = tf.keras.layers.RNN(cell, return_sequences=True)

    rnn_output_eager_seq = rnn_seq(inputs_batch)
    rnn_output_symbolic_seq = rnn_seq(inputs_batch_symbolic)

    assert rnn_output_eager_seq.shape.as_list() == [batch_size, seq_len, units]
    assert rnn_output_symbolic_seq.shape.as_list() == [None, None, units]
