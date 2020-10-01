from unittest import TestCase

import numpy as np
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import RNN, TimeDistributed, Dense, Concatenate

from data_preparation import get_datasets
from graph_reduce_cell import GraphReduceCell
from subgraphing import get_subgraphs, Subgraphing


def prepare_matrices():
    adj_matrix = np.array([[0, 1, 0, 0],
                           [1, 0, 1, 1],
                           [0, 1, 0, 1],
                           [0, 1, 1, 0]])

    no_bond = np.zeros((4, 4, 3))
    no_bond[:, :, 0] = 1

    fosforowe = np.zeros((4, 4, 3))
    fosforowe[:, :, 1] = 1

    wodorowe = np.zeros((4, 4, 3))
    wodorowe[:, :, 2] = 1

    broadcasted_adj_matrix = np.repeat(adj_matrix[:, :, np.newaxis], 3, axis=2)

    x = np.where(broadcasted_adj_matrix, fosforowe, no_bond)

    wodorowe_chooser = np.array([[0, 0, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 0, 0],
                                 [0, 1, 0, 0]])
    broadcasted_wodorowe_chooser = np.repeat(wodorowe_chooser[:, :, np.newaxis], 3, axis=2)

    edges_features_matrix = np.where(broadcasted_wodorowe_chooser, wodorowe, x)

    features_seq = (np.random.rand(4, 14) < 0.5).astype(np.float32)

    return features_seq, adj_matrix, edges_features_matrix


def get_exp_dataset(batch_size):
    train_valid_ds, public_test_ds, private_test_ds = get_datasets()
    exp_ds = train_valid_ds.take(batch_size).batch(batch_size)
    return exp_ds


def get_test_model(seq_len, stacked_features_size, edges_features_matrix_depth, neighbourhood_size, units):
    # INPUTS
    stacked_base_features_input = Input(shape=(seq_len, stacked_features_size), name='stacked_base_features')
    adjacency_matrix_input = Input(shape=(seq_len, seq_len), name='adjacency_matrix')
    edges_features_matrix_input = Input(shape=(seq_len, seq_len, edges_features_matrix_depth),
                                        name='edges_features_matrix')
    inputs = (stacked_base_features_input, adjacency_matrix_input, edges_features_matrix_input)

    # ACTUAL MODEL
    x = Subgraphing(neighbourhood_size)(inputs)
    x = RNN(GraphReduceCell(units), return_sequences=True)(x)

    # OUTPUTS
    reactivity_pred = TimeDistributed(Dense(1), name='reactivity')(x)
    deg_Mg_pH10_pred = TimeDistributed(Dense(1), name='deg_Mg_pH10')(x)
    deg_Mg_50C_pred = TimeDistributed(Dense(1), name='deg_Mg_50C')(x)

    scored_outputs = [reactivity_pred, deg_Mg_pH10_pred, deg_Mg_50C_pred]
    stacked_outputs = Concatenate(axis=2, name='stacked_outputs')(scored_outputs)

    # MODEL DEFINING
    model = Model(inputs=inputs, outputs={'stacked_scored_labels': stacked_outputs}, name='graph_reduce_model')

    return model

def prepare_control_matrices():
    control_subgraphs_adjacency_matrices = [
        np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]]),
        np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]]),
        np.array([[0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]]),
        np.array([[0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]])
    ]

    control_subgraphs_edges_features_matrices = [
        np.array([[[1, 0, 0], [0, 1, 0], [1, 0, 0]],
                  [[0, 1, 0], [1, 0, 0], [0, 1, 0]],
                  [[1, 0, 0], [0, 1, 0], [1, 0, 0]]]),
        np.array([[[1, 0, 0], [0, 1, 0], [1, 0, 0]],
                  [[0, 1, 0], [1, 0, 0], [0, 1, 0]],
                  [[1, 0, 0], [0, 1, 0], [1, 0, 0]]]),
        np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                  [[0, 1, 0], [1, 0, 0], [0, 1, 0]],
                  [[0, 0, 1], [0, 1, 0], [1, 0, 0]]]),
        np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                  [[0, 1, 0], [1, 0, 0], [0, 1, 0]],
                  [[0, 0, 1], [0, 1, 0], [1, 0, 0]]]),
    ]

    return control_subgraphs_adjacency_matrices, control_subgraphs_edges_features_matrices


class TestGraphReduceModel(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features_seq, cls.adj_matrix, cls.edges_features_matrix = prepare_matrices()
        cls.control_subgraphs_adjacency_matrices, cls.control_subgraphs_edges_features_matrices = prepare_control_matrices()

        cls.BATCH_SIZE = 64
        cls.SEQ_LEN = 68
        cls.STACKED_FEATURES_SIZE = 14
        cls.EDGES_FEATURES_MATRIX_DEPTH = 3
        cls.NEIGHBOURHOOD_SIZE = 10
        cls.UNITS = 7
        cls.STACKED_LABELS_SIZE = 3

        cls.exp_ds = get_exp_dataset(cls.BATCH_SIZE)
        cls.test_model = get_test_model(cls.SEQ_LEN, cls.STACKED_FEATURES_SIZE, cls.EDGES_FEATURES_MATRIX_DEPTH,
                                        cls.NEIGHBOURHOOD_SIZE, cls.UNITS)

    def test_get_subgraphs(self):
        subgraphs_features_sequences, subgraphs_adjacency_matrices, subgraphs_edges_features_matrices = get_subgraphs(
            self.features_seq, self.adj_matrix, self.edges_features_matrix, 3)

        for seq in subgraphs_features_sequences:
            self.assertEqual(list(seq.shape), [3, 14])

        for matrix, control_matrix in zip(subgraphs_adjacency_matrices,
                                          self.control_subgraphs_adjacency_matrices):
            self.assertTrue(np.array_equal(matrix, control_matrix))

        for matrix, control_matrix in zip(subgraphs_edges_features_matrices,
                                          self.control_subgraphs_edges_features_matrices):
            self.assertTrue(np.array_equal(matrix, control_matrix))

    def test_subgraphing(self):
        data_batch = next(iter(self.exp_ds))
        x, y = data_batch
        inputs_batch = (x['stacked_base_features'], x['adjacency_matrix'], x['edges_features_matrix'])

        self.assertEqual(inputs_batch[0].shape, (self.BATCH_SIZE, self.SEQ_LEN, self.STACKED_FEATURES_SIZE))
        self.assertEqual(inputs_batch[1].shape, (self.BATCH_SIZE, self.SEQ_LEN, self.SEQ_LEN))
        self.assertEqual(inputs_batch[2].shape, (self.BATCH_SIZE, self.SEQ_LEN, self.SEQ_LEN,
                                                 self.EDGES_FEATURES_MATRIX_DEPTH))

        outputs_batch = Subgraphing(10)(inputs_batch)

        self.assertEqual(type(outputs_batch), tuple)
        self.assertEqual(len(outputs_batch), 3)

        self.assertEqual(outputs_batch[0].shape, [self.BATCH_SIZE, self.SEQ_LEN, self.NEIGHBOURHOOD_SIZE,
                                                  self.STACKED_FEATURES_SIZE])
        self.assertEqual(outputs_batch[1].shape, [self.BATCH_SIZE, self.SEQ_LEN, self.NEIGHBOURHOOD_SIZE,
                                                  self.NEIGHBOURHOOD_SIZE])
        self.assertEqual(outputs_batch[2].shape, [self.BATCH_SIZE, self.SEQ_LEN, self.NEIGHBOURHOOD_SIZE,
                                                  self.NEIGHBOURHOOD_SIZE, self.EDGES_FEATURES_MATRIX_DEPTH])

    def test_graph_reduce_model(self):
        predictions = self.test_model.predict(self.exp_ds)

        self.assertEqual(type(predictions), dict)

        keys = list(predictions.keys())
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0], 'stacked_scored_labels')

        self.assertEqual(list(predictions['stacked_scored_labels'].shape), [self.BATCH_SIZE, self.SEQ_LEN,
                                                                            self.STACKED_LABELS_SIZE])
