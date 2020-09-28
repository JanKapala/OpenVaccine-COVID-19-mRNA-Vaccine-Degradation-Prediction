import tensorflow as tf
import numpy as np
from bidict import bidict

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.models import Model

from data_preparation import get_datasets, only_stacked_scored_labels


class WindowingLayer(tf.keras.layers.Layer):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def call(self, inputs, **kwargs):
        batch_size, seq_len, feature_size = inputs.shape

        first_part = tf.stack([inputs[:, i: i + self.window_size] for i in range(0, seq_len - self.window_size + 1)],
                              axis=1)
        incomplete_part = [inputs[:, i: i + self.window_size] for i in range(seq_len - self.window_size + 1, seq_len)]

        second_part = np.zeros([batch_size, self.window_size - 1, self.window_size, feature_size])

        for i in range(self.window_size - 1):
            for j in range(self.window_size - i - 1):
                for k in range(batch_size):
                    second_part[k, i, j] = incomplete_part[i][k][j]

        return tf.concat([first_part, second_part], axis=1)


def get_neighbours(adj_matrix, neigh_size, vertex_id):
    processed = []
    in_progress = [vertex_id]

    while len(processed) < neigh_size + 1:
        processing = in_progress.pop(0)
        neighbours = np.where(adj_matrix[processing])[1]
        for v in neighbours:
            if not (v in processed or v in in_progress):
                in_progress.append(v)

        processed.append(processing)

    return sorted(processed)


def get_subfeatures(features, adj_matrix, edges_matrix, neigh_size):
    windowed_adj, windowed_edges, windowed_features = [], [], []

    for vertex_id in range(adj_matrix.shape[0]):
        neighbours = get_neighbours(adj_matrix, neigh_size-1, vertex_id)

        new_adj = np.copy(adj_matrix[neighbours][:, neighbours])
        new_edges = np.copy(edges_matrix[neighbours][:, neighbours])
        new_features = np.copy(features[neighbours])

        windowed_adj.append(new_adj)
        windowed_edges.append(new_edges)
        windowed_features.append(new_features)

    windowed_adj = np.stack(windowed_adj, axis=0)
    windowed_edges = np.stack(windowed_edges, axis=0)
    windowed_features = np.stack(windowed_features, axis=0)

    return windowed_adj, windowed_edges, windowed_features


def windowing_layer(features_batch, adj_matrix_batch, edges_matrix_batch, neigh_size):
    batch_size = adj_matrix_batch.shape[0]

    batch_windowed_adj, batch_windowed_edges, batch_windowed_features = [], [], []

    for example in range(batch_size):
        example_windowed_adj, example_windowed_edges, example_windowing_features = get_subfeatures(
            adj_matrix_batch[example], edges_matrix_batch[example], features_batch[example], neigh_size)
        batch_windowed_adj.append(example_windowed_adj)
        batch_windowed_edges.append(example_windowed_edges)
        batch_windowed_features.append(example_windowing_features)

    batch_windowed_adj = np.stack(batch_windowed_adj, axis=0)
    batch_windowed_edges = np.stack(batch_windowed_edges, axis=0)
    batch_windowed_features = np.stack(batch_windowed_features, axis=0)

    return batch_windowed_adj, batch_windowed_edges, batch_windowed_features


class SubgraphingLayer(tf.keras.layers.Layer):
    def __init__(self, neighbourhood_size, **kwargs):
        super().__init__(**kwargs)
        self.neighbourhood_size = neighbourhood_size

    def call(self, inputs, training=None, mask=None):
        return windowing_layer(inputs[0], inputs[1], inputs[2], self.neighbourhood_size)


if __name__ == '__main__':
    # Get sample data batch
    train_valid_ds, public_test_ds, private_test_ds = get_datasets()
    train_valid_with_stacked_labels_ds = train_valid_ds.map(only_stacked_scored_labels)
    exp_ds = train_valid_with_stacked_labels_ds.batch(2).take(1)
    batch = next(iter(exp_ds))

    # Define inputs
    INPUT_SEQUENCE_LENGTH = None
    EDGES_FEATURES_MATRIX_DEPTH = 3
    sequence_input = Input(shape=(INPUT_SEQUENCE_LENGTH, 4),
                           name='sequence')
    structure_input = Input(shape=(INPUT_SEQUENCE_LENGTH, 3),
                            name='structure')
    predicted_loop_type_input = Input(shape=(INPUT_SEQUENCE_LENGTH, 7),
                                      name='predicted_loop_type')

    adjacency_matrix_input = Input(shape=(None, None), name='adjacency_matrix')
    edges_features_matrix_input = Input(shape=(None, None, EDGES_FEATURES_MATRIX_DEPTH), name='edges_features_matrix')

    inputs = [sequence_input, structure_input, predicted_loop_type_input, adjacency_matrix_input,
              edges_features_matrix_input]

    # Stack features
    features_to_stack = [sequence_input, structure_input, predicted_loop_type_input]
    features_input = Concatenate(axis=2, name='input_stacking_layer')(features_to_stack)

    # Prepare input for subgraphing layer
    input_for_sg = (adjacency_matrix_input, edges_features_matrix_input, features_input)

    sg_layer = SubgraphingLayer(10)
    outputs = sg_layer(input_for_sg)

    # Create model
    exp_model = Model(inputs=inputs, outputs=outputs)

    # Test model with subgraphing layer
    print(exp_model(batch))
