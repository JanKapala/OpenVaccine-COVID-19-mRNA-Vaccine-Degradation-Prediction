import tensorflow as tf
import numpy as np
from bidict import bidict

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.models import Model

from data_preparation import get_datasets, only_stacked_scored_labels

def get_neighbours(adj_matrix, neigh_size, vertex_id):
    processed = []
    in_progress = [vertex_id]

    while len(processed) < neigh_size + 1:
        processing = in_progress.pop(0)
        neighbours = np.where(adj_matrix[processing])[0]
        for v in neighbours:
            if not (v in processed or v in in_progress):
                in_progress.append(v)

        processed.append(processing)

    return sorted(processed)


def get_subfeatures(features, adj_matrix, edges_matrix, neigh_size):

    windowed_adj, windowed_edges, windowed_features = [], [], []
    all_neighbours = np.array([sorted(list(distances.keys())[:neigh_size]) for node, distances in list(bellmann_ford.items())])

    for vertex_id in range(adj_matrix.shape[0]):
        neighbours = all_neighbours[vertex_id]
        
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
    features_batch = features_batch.numpy()
    adj_matrix_batch = adj_matrix_batch.numpy()
    edges_matrix_batch = edges_matrix_batch.numpy()
    
    batch_size = tf.shape(adj_matrix_batch)[0]

    batch_windowed_adj, batch_windowed_edges, batch_windowed_features = [], [], []

    for example in range(batch_size):
        example_windowed_adj, example_windowed_edges, example_windowing_features = get_subfeatures(
            features_batch[example], adj_matrix_batch[example], edges_matrix_batch[example], neigh_size)
        batch_windowed_adj.append(example_windowed_adj)
        batch_windowed_edges.append(example_windowed_edges)
        batch_windowed_features.append(example_windowing_features)

    batch_windowed_adj = np.stack(batch_windowed_adj, axis=0)
    batch_windowed_edges = np.stack(batch_windowed_edges, axis=0)
    batch_windowed_features = np.stack(batch_windowed_features, axis=0)

    return batch_windowed_features, batch_windowed_adj, batch_windowed_edges


class SubgraphingLayer(tf.keras.layers.Layer):
    def __init__(self, neighbourhood_size, **kwargs):
        super().__init__(**kwargs)
        self.neighbourhood_size = neighbourhood_size

    def call(self, inputs, training=None, mask=None):
        features_batch = inputs[0]
        adj_matrix_batch = inputs[1]
        edges_features_matrix_batch = inputs[2]

        outputs = tf.py_function(func=windowing_layer,
                                 inp=[features_batch, adj_matrix_batch, edges_features_matrix_batch,
                                      self.neighbourhood_size],
                                 Tout=[tf.float32, tf.float32, tf.float32,])

#         Maybe some outputs.set_shape(...)

        return outputs


if __name__ == '__main__':
    # GET SMALL EXPERIMENTAL DATASET
    # WARRING: It takes a while, so I suggest to move this testing code to jupyter (split code into cells and use `from custom_layers import *`)
    train_valid_ds, public_test_ds, private_test_ds = get_datasets()
    train_valid_with_stacked_labels_ds = train_valid_ds.map(only_stacked_scored_labels)
    exp_ds = train_valid_with_stacked_labels_ds.batch(2).take(1)

    # BUILD TESTING MODEL

    # Define inputs
    sequence_input = Input(shape=(None, 4), name='sequence')
    structure_input = Input(shape=(None, 3), name='structure')
    predicted_loop_type_input = Input(shape=(None, 7), name='predicted_loop_type')

    adjacency_matrix_input = Input(shape=(None, None), name='adjacency_matrix')
    edges_features_matrix_input = Input(shape=(None, None, 3), name='edges_features_matrix')

    inputs = [sequence_input, structure_input, predicted_loop_type_input, adjacency_matrix_input,
              edges_features_matrix_input]

    # Stack features
    features_to_stack = [sequence_input, structure_input, predicted_loop_type_input]
    features_input = Concatenate(axis=2, name='features')(features_to_stack)

    # Prepare input for subgraphing layer
    inputs_for_sg_layer = [features_input, adjacency_matrix_input, edges_features_matrix_input]

    sg_layer = SubgraphingLayer(10)
    outputs = sg_layer(inputs_for_sg_layer)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    output = model.predict(exp_ds)

    # TODO: some check on output
