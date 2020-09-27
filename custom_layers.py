import tensorflow as tf
import numpy as np
from bidict import bidict


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

    return processed


def get_subfeatures(adj_matrix, edges_matrix, features, neigh_size):
    windowed_adj, windowed_edges, windowed_features = [], [], []

    for vertex_id in range(adj_matrix.shape[0]):
        neighbours = get_neighbours(adj_matrix, neigh_size, vertex_id)
        mapped_orig_dict = bidict({i: j for i, j in enumerate(neighbours)})

        new_adj = np.zeros((len(neighbours), len(neighbours)))
        new_edges = np.zeros((len(neighbours), len(neighbours), edges_matrix.shape[-1]))
        new_features = np.copy(features[neighbours])

        for mapped_v in mapped_orig_dict.keys():
            for orig_v in np.where(adj_matrix[mapped_orig_dict[mapped_v]])[1]:
                if orig_v in list(mapped_orig_dict.values()):
                    new_adj[mapped_v][mapped_orig_dict.inverse[orig_v]] = adj_matrix[mapped_orig_dict[mapped_v], orig_v]
                    new_edges[mapped_v][mapped_orig_dict.inverse[orig_v]] = edges_matrix[
                        mapped_orig_dict[mapped_v], orig_v]

        windowed_adj.append(new_adj)
        windowed_edges.append(new_edges)
        windowed_features.append(new_features)

    windowed_adj = np.stack(windowed_adj, axis=0)
    windowed_edges = np.stack(windowed_edges, axis=0)
    windowed_features = np.stack(windowed_features, axis=0)

    return windowed_adj, windowed_edges, windowed_features


def windowing_layer(adj_matrix_batch, edges_matrix_batch, features_batch, neigh_size):
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

    def call(self, inputs):
        return windowing_layer(inputs[0], inputs[1], inputs[2])
