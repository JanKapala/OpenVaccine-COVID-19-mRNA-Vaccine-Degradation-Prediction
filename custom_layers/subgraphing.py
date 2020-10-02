import tensorflow as tf
import numpy as np

import networkx as nx
from networkx.algorithms.shortest_paths.weighted import all_pairs_dijkstra_path_length as dijkstra


def get_neighbourhood_indices(adjacency_matrix, neigh_size):
    # construct nx Graph
    G = nx.from_numpy_matrix(adjacency_matrix)

    # extract neigh_size neighbours
    shortest_paths_results = dict(dijkstra(G))

    neighbourhood_indices = [sorted(list(distances.keys())[:neigh_size]) for node, distances in
                             list(shortest_paths_results.items())]

    # Without this slice will be NOT get correctly
    neighbourhood_indices = np.array(neighbourhood_indices)

    return neighbourhood_indices


def extract_subgraphs_features(features_sequence, neighbourhood_indices):
    # Choose rows by indices
    subgraphs_features = features_sequence[neighbourhood_indices]

    return subgraphs_features


def extract_subgraphs_adj_matrices(matrix, neighbourhood_indices):
    # Get final dimension of each subgraph's matrix
    N = matrix.shape[0]
    n = neighbourhood_indices.shape[1]

    # Choose rows by indices
    x = matrix[neighbourhood_indices]

    # Transpose to make possible choosing columns
    x = np.transpose(x, (0, 2, 1))

    # Choose columns by indices
    x = x[:, neighbourhood_indices]

    # Choose diagonal
    diagonal_indices = np.arange(0, N ** 2, N + 1)
    x = x.reshape(N ** 2, n, n)[diagonal_indices]

    # Subgraphs' matrices need transposing
    subgraphs_matrices = np.transpose(x, (0, 2, 1))

    return subgraphs_matrices


def extract_subgraphs_edge_features_matrices(matrix, neighbourhood_indices):
    # Get final dimension of each subgraph's matrix
    N = matrix.shape[0]
    n = neighbourhood_indices.shape[1]

    # Choose rows by indices
    x = matrix[neighbourhood_indices]

    # Transpose to make possible choosing columns
    x = np.transpose(x, (0, 2, 1, 3))

    # Choose columns by indices
    x = x[:, neighbourhood_indices]

    # Choose diagonal
    diagonal_indices = np.arange(0, N ** 2, N + 1)
    x = x.reshape(N ** 2, n, n, 3)[diagonal_indices]

    # Subgraphs' matrices need transposing
    subgraphs_matrices = np.transpose(x, (0, 2, 1, 3))

    return subgraphs_matrices


def get_subgraphs(features_sequence, adjacency_matrix, edges_features_matrix, neigh_size):
    neighbourhood_indices = get_neighbourhood_indices(adjacency_matrix, neigh_size)

    subgraphs_features_sequences = extract_subgraphs_features(features_sequence, neighbourhood_indices)
    subgraphs_adjacency_matrices = extract_subgraphs_adj_matrices(adjacency_matrix, neighbourhood_indices)
    subgraphs_edges_features_matrices = extract_subgraphs_edge_features_matrices(edges_features_matrix,
                                                                                 neighbourhood_indices)

    return subgraphs_features_sequences, subgraphs_adjacency_matrices, subgraphs_edges_features_matrices


def get_subgraphs_of_batch(features_batch, adj_matrix_batch, edges_features_matrix_batch, neigh_size):
    features_batch = features_batch.numpy().astype(np.float32)
    adj_matrix_batch = adj_matrix_batch.numpy().astype(np.float32)
    edges_features_matrix_batch = edges_features_matrix_batch.numpy().astype(np.float32)

    batch_size = features_batch.shape[0]

    subgraphed_features_batch = []
    subgraphed_adj_matrices_batch = []
    subgraphed_edges_features_matrices_batch = []

    for i in range(batch_size):
        subgraphs_features_sequences, subgraphs_adjacency_matrices, subgraphs_edges_features_matrices = get_subgraphs(
            features_batch[i], adj_matrix_batch[i], edges_features_matrix_batch[i], neigh_size)
        subgraphed_features_batch.append(subgraphs_features_sequences)
        subgraphed_adj_matrices_batch.append(subgraphs_adjacency_matrices)
        subgraphed_edges_features_matrices_batch.append(subgraphs_edges_features_matrices)

    subgraphed_features_batch = np.stack(subgraphed_features_batch, axis=0)
    subgraphed_adj_matrices_batch = np.stack(subgraphed_adj_matrices_batch, axis=0)
    subgraphed_edges_features_matrices_batch = np.stack(subgraphed_edges_features_matrices_batch, axis=0)

    return subgraphed_features_batch, subgraphed_adj_matrices_batch, subgraphed_edges_features_matrices_batch


class Subgraphing(tf.keras.layers.Layer):
    def __init__(self, neighbourhood_size, **kwargs):
        super().__init__(**kwargs)
        self.neighbourhood_size = neighbourhood_size

    def call(self, inputs, training=None, mask=None):
        features_batch = inputs[0]
        adj_matrix_batch = inputs[1]
        edges_features_matrix_batch = inputs[2]

        outputs = tf.py_function(func=get_subgraphs_of_batch,
                                 inp=[features_batch, adj_matrix_batch, edges_features_matrix_batch,
                                      self.neighbourhood_size],
                                 Tout=[tf.float32, tf.float32, tf.float32])

        subgraphed_features_batch, subgraphed_adj_matrices_batch, subgraphed_edges_features_matrices_batch = outputs

        batch_size = None
        seq_len = None
        features_size = features_batch.shape[2]

        shape = [batch_size, seq_len, self.neighbourhood_size]

        s1 = shape + [features_size]
        s2 = shape + [self.neighbourhood_size]
        s3 = shape + [self.neighbourhood_size, 3]

        subgraphed_features_batch.set_shape(s1)
        subgraphed_adj_matrices_batch.set_shape(s2)
        subgraphed_edges_features_matrices_batch.set_shape(s3)

        return tuple(outputs)

    def get_config(self):
        config = super().get_config()
        config.update({'neighbourhood_size': self.neighbourhood_size})
        return config
