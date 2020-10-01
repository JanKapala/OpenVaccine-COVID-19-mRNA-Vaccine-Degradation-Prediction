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

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units,
                       'conv_graph_layer': self.conv_graph_layer,
                       'pool_graph_layer': self.pool_graph_layer})
        return config