import os

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow import TensorSpec

from custom_layers.subgraphing import Subgraphing

from visualization import *

# CONSTANTS
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

DATASETS_DIR = os.path.join(SCRIPT_DIR, 'datasets')
RAW_TRAIN_DS_PATH = os.path.join(DATASETS_DIR, 'train.json')
RAW_TEST_DS_PATH = os.path.join(DATASETS_DIR, 'test.json')
TRAIN_VALID_DS_PATH = os.path.join(DATASETS_DIR, 'train_valid_ds')
PUBLIC_TEST_DS_PATH = os.path.join(DATASETS_DIR, 'public_test_ds')
PRIVATE_TEST_DS_PATH = os.path.join(DATASETS_DIR, 'private_test_ds')
SUBGRAPHED_DATASETS_DIR = os.path.join(DATASETS_DIR, 'subgraphed')

SUBMISSIONS_DIR = os.path.join(SCRIPT_DIR, 'submissions')
SAMPLE_SUBMISSION_PATH = os.path.join(SUBMISSIONS_DIR, 'sample_submission.csv')

FEATURE_NAMES = ['sequence', 'structure', 'predicted_loop_type', 'adjacency_matrix', 'edges_features_matrix',
                 'seq_scored']
ERROR_LABEL_NAMES = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C']
NORMAL_LABEL_NAMES = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
ALL_LABEL_NAMES = ERROR_LABEL_NAMES + NORMAL_LABEL_NAMES
SCORED_LABEL_NAMES = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']

SUBGRAPH_OPERATION_BATCH_SIZE=64

TEST_DS_SPEC = ({'sequence': TensorSpec(shape=(None, 4), dtype=tf.float32, name=None),
                 'structure': TensorSpec(shape=(None, 3), dtype=tf.float32, name=None),
                 'predicted_loop_type': TensorSpec(shape=(None, 7), dtype=tf.float32, name=None),
                 'adjacency_matrix': TensorSpec(shape=(None, None), dtype=tf.float32, name=None),
                 'edges_features_matrix': TensorSpec(shape=(None, None, 3), dtype=tf.float32, name=None),
                 'seq_scored': TensorSpec(shape=(), dtype=tf.float32, name=None),
                 'stacked_base_features': TensorSpec(shape=(None, 14), dtype=tf.float32, name=None)})

TRAIN_DS_SPEC = (TEST_DS_SPEC, {'stacked_scored_labels': TensorSpec(shape=(None, 3), dtype=tf.float32, name=None)})


def _get_subgraphed_test_ds_spec(neighbourhood_size):
    return ({'sequence': TensorSpec(shape=(None, 4), dtype=tf.float32, name=None),
             'structure': TensorSpec(shape=(None, 3), dtype=tf.float32, name=None),
             'predicted_loop_type': TensorSpec(shape=(None, 7), dtype=tf.float32, name=None),
             'adjacency_matrix': TensorSpec(shape=(None, neighbourhood_size, neighbourhood_size), dtype=tf.float32,
                                            name=None),
             'edges_features_matrix': TensorSpec(shape=(None, neighbourhood_size, neighbourhood_size, 3),
                                                 dtype=tf.float32, name=None),
             'seq_scored': TensorSpec(shape=(), dtype=tf.float32, name=None),
             'stacked_base_features': TensorSpec(shape=(None, neighbourhood_size, 14), dtype=tf.float32, name=None)})


def _get_subgraphed_train_ds_spec(neighbourhood_size):
    return _get_subgraphed_test_ds_spec(neighbourhood_size), {
        'stacked_scored_labels': TensorSpec(shape=(None, 3), dtype=tf.float32, name=None)}


# HELPER PREP FUNCTIONS
def _struct2matrices(structure):
    n = len(structure)
    adjacency_matrix = np.zeros((n, n))
    edges_features_matrix = np.zeros((n, n, 3))
    edges_features_matrix[:, :, 0] = 1  # Initialize with [1, 0, 0]

    def add_bond(i, j, bond_type):
        bond_map = {'phosphodiester': [0, 1, 0], 'hydrogen': [0, 0, 1]}
        bond = bond_map[bond_type]

        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
        edges_features_matrix[i, j] = bond
        edges_features_matrix[j, i] = bond

    stack = []
    for i, symbol in enumerate(structure):
        if i != 0:
            add_bond(i, i - 1, 'phosphodiester')

        if symbol == '.':
            continue
        elif symbol == '(':
            stack.append(i)
        elif symbol == ')':
            j = stack.pop()
            add_bond(i, j, 'hydrogen')
    return adjacency_matrix, edges_features_matrix


def _add_graph_matrices_columns(raw_ds):
    raw_ds.insert(4, 'adjacency_matrix', None)
    raw_ds.insert(5, 'edges_features_matrix', None)
    temp = raw_ds['structure'].map(lambda structure: _struct2matrices(structure))
    raw_ds[['adjacency_matrix', 'edges_features_matrix']] = pd.DataFrame(temp.tolist())


def _create_train_valid_ds(raw_train_valid_ds, feature_names, all_label_names):
    # Transform dataframe using encoders
    preprocessing = make_column_transformer(
        (make_pipeline(_FeaturesEncoder()), feature_names),
        (make_pipeline(_LabelsEncoder(all_label_names)), all_label_names),
    )

    data = pd.DataFrame(preprocessing.fit_transform(raw_train_valid_ds))
    data.columns = feature_names + all_label_names

    # Split into features and labels and convert to dictionaries of tensors
    features_dict = data[feature_names].to_dict(orient='list')
    for column, values in features_dict.items():
        features_dict[column] = tf.constant(values, dtype=tf.float32)

    labels_dict = data[all_label_names].to_dict(orient='list')
    for column, values in labels_dict.items():
        labels_dict[column] = tf.constant(values, dtype=tf.float32)

    # Create tf.data.Dataset
    train_valid_ds = tf.data.Dataset.from_tensor_slices((features_dict, labels_dict))

    return train_valid_ds


def _create_test_ds(raw_test_ds, feature_names):
    # Transform dataframe using encoders
    preprocessing = make_column_transformer((make_pipeline(_FeaturesEncoder()), feature_names))

    data = pd.DataFrame(preprocessing.fit_transform(raw_test_ds))
    data.columns = feature_names

    # Split into features and labels and convert to dictionaries of tensors
    features_dict = data[feature_names].to_dict(orient='list')
    for column, values in features_dict.items():
        features_dict[column] = tf.constant(values, dtype=tf.float32)

    # Create tf.data.Dataset
    test_ds = tf.data.Dataset.from_tensor_slices(features_dict)

    return test_ds


def _trim_unlabeled_features(x):
    label_seq_len = tf.cast(x['seq_scored'], tf.int32)
    seq_features = ['sequence', 'structure', 'predicted_loop_type']

    for seq_feature in seq_features:
        x[seq_feature] = x[seq_feature][:label_seq_len, :]

    x['adjacency_matrix'] = x['adjacency_matrix'][:label_seq_len, :label_seq_len]
    x['edges_features_matrix'] = x['edges_features_matrix'][:label_seq_len, :label_seq_len, :]

    return x


def _only_stacked_scored_labels(y):
    tensors_to_stack = [y[scored_label] for scored_label in SCORED_LABEL_NAMES]
    stacked_scored_labels = tf.stack(tensors_to_stack, axis=1)
    y = {'stacked_scored_labels': stacked_scored_labels}
    return y


def _add_stacked_base_features(x):
    tensors_to_concat = [x[feature_name] for feature_name in ['sequence', 'structure', 'predicted_loop_type']]
    stacked_base_features = tf.concat(tensors_to_concat, axis=1)
    x['stacked_base_features'] = stacked_base_features
    return x


def _subgraph_dataset(ds, neighbourhood_size):
    def subgraphing_map_fn(*args):
        if len(args) == 2:
            x, y = args
        else:
            x = args[0]

        base_inputs = x['stacked_base_features']
        adjacency_matrix_inputs = x['adjacency_matrix']
        edges_features_matrix_inputs = x['edges_features_matrix']
        inputs = (base_inputs, adjacency_matrix_inputs, edges_features_matrix_inputs)

        subgraphed = Subgraphing(neighbourhood_size)(inputs)

        subgraphed_base_inputs, subgraphed_adjacency_matrix_inputs, subgraphed_edges_features_matrix_inputs = subgraphed
        x['stacked_base_features'] = subgraphed_base_inputs
        x['adjacency_matrix'] = subgraphed_adjacency_matrix_inputs
        x['edges_features_matrix'] = subgraphed_edges_features_matrix_inputs

        if len(args) == 2:
            return x, y
        else:
            return x

    return ds.batch(SUBGRAPH_OPERATION_BATCH_SIZE).map(subgraphing_map_fn).unbatch()
    

def _save_dataset(ds, path):
    # Get dataset spec - needed for loading the dataset
    ds_spec = tf.data.DatasetSpec.from_value(ds)._element_spec

    # Save the dataset
    tf.data.experimental.save(ds, path)

    # Return the dataset_spec just in case
    return ds_spec


def _load_dataset(path, ds_spec):
    return tf.data.experimental.load(path, ds_spec)


# HELPER PREP CLASSES
class _LabelsEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, labels):
        self.labels = labels

    def fit(self, X, y=None, *args, **kwargs):
        return self

    def transform(self, X, *args, **kwargs):
        X = X.apply(self.transform_example, axis='columns')
        return X

    def transform_example(self, example):
        for label in self.labels:
            example[label] = np.array(example[label], dtype=np.float32)
        return example


class _FeaturesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.base2vec = {'A': [1, 0, 0, 0],
                         'C': [0, 1, 0, 0],
                         'G': [0, 0, 1, 0],
                         'U': [0, 0, 0, 1]}
        self.structpart2vec = {'.': [1, 0, 0],
                               '(': [0, 1, 0],
                               ')': [0, 0, 1]}
        self.looptype2vec = {'S': [1, 0, 0, 0, 0, 0, 0],
                             'M': [0, 1, 0, 0, 0, 0, 0],
                             'I': [0, 0, 1, 0, 0, 0, 0],
                             'B': [0, 0, 0, 1, 0, 0, 0],
                             'H': [0, 0, 0, 0, 1, 0, 0],
                             'E': [0, 0, 0, 0, 0, 1, 0],
                             'X': [0, 0, 0, 0, 0, 0, 1]}

    def fit(self, X, y=None, *args, **kwargs):
        return self

    def transform(self, X, *args, **kwargs):
        X = X.apply(self.transform_example, axis='columns', result_type='expand')
        return X

    def transform_example(self, example):
        encoded_sequence = [self.base2vec[base] for base in example['sequence']]
        example['sequence'] = encoded_sequence

        encoded_structure = [self.structpart2vec[structpart] for structpart in example['structure']]
        example['structure'] = encoded_structure

        encoded_loop_type = [self.looptype2vec[looptype] for looptype in example['predicted_loop_type']]
        example['predicted_loop_type'] = encoded_loop_type

        return example


# PUBLIC FUNCTIONS
def get_raw_datasets():
    raw_train_valid_ds = pd.read_json(RAW_TRAIN_DS_PATH, lines=True)
    raw_test_ds = pd.read_json(RAW_TEST_DS_PATH, lines=True)

    _add_graph_matrices_columns(raw_train_valid_ds)
    _add_graph_matrices_columns(raw_test_ds)

    raw_public_test_ds = raw_test_ds.loc[raw_test_ds['seq_length'] == 107]
    raw_private_test_ds = raw_test_ds.loc[raw_test_ds['seq_length'] == 130]

    return raw_train_valid_ds, raw_public_test_ds, raw_private_test_ds


def convert_to_datasets(raw_train_valid_ds, raw_public_test_ds, raw_private_test_ds, trim=True):
    train_valid_ds = _create_train_valid_ds(raw_train_valid_ds, FEATURE_NAMES, ALL_LABEL_NAMES)
    public_test_ds = _create_test_ds(raw_public_test_ds, FEATURE_NAMES)
    private_test_ds = _create_test_ds(raw_private_test_ds, FEATURE_NAMES)

    if trim:
        train_valid_ds = train_valid_ds.map(lambda x, y: (_trim_unlabeled_features(x), y))
        public_test_ds = public_test_ds.map(_trim_unlabeled_features)
        private_test_ds = private_test_ds.map(_trim_unlabeled_features)

    train_valid_ds = train_valid_ds.map(lambda x, y: (x, _only_stacked_scored_labels(y)))

    train_valid_ds = train_valid_ds.map(lambda x, y: (_add_stacked_base_features(x), y))
    public_test_ds = public_test_ds.map(_add_stacked_base_features)
    private_test_ds = private_test_ds.map(_add_stacked_base_features)

    return train_valid_ds, public_test_ds, private_test_ds

def ds_summary(ds, name):
    print(f"{name}")
    print(ds)
    print(f"Cardinality: {ds.cardinality()}")
    print('\n')
    print(f"{name} example")
    print(next(iter(ds)))


def split_into_train_and_valid(train_valid_ds, split_factor=0.3):    
    length = 0
    for i in train_valid_ds:
        length +=1
        
    valid_ds_length = int(split_factor * length)
    train_valid_ds_shuffled = train_valid_ds.shuffle(length)
    train_ds = train_valid_ds_shuffled.skip(valid_ds_length)
    valid_ds = train_valid_ds_shuffled.take(valid_ds_length)
    return train_ds, valid_ds


def trim(x, y, trim_to):
    for label in x.keys():
        x[label] = x[label][:trim_to, :]
    for label in y.keys():
        y[label] = y[label][:trim_to, :]
    return x, y


def get_sample_submission():
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    return sample_submission


def load_base_datasets(save=True):
    if os.path.isdir(TRAIN_VALID_DS_PATH):
        train_valid_ds = _load_dataset(TRAIN_VALID_DS_PATH, TRAIN_DS_SPEC)
        public_test_ds = _load_dataset(PUBLIC_TEST_DS_PATH, TEST_DS_SPEC)
        private_test_ds = _load_dataset(PRIVATE_TEST_DS_PATH, TEST_DS_SPEC)

        return train_valid_ds, public_test_ds, private_test_ds
    
    raw_train_valid_ds, raw_public_test_ds, raw_private_test_ds = get_raw_datasets()
    
    train_valid_ds, public_test_ds, private_test_ds = convert_to_datasets(raw_train_valid_ds, raw_public_test_ds, raw_private_test_ds)
    
    if save:
        _save_dataset(train_valid_ds, TRAIN_VALID_DS_PATH)
        _save_dataset(public_test_ds, PUBLIC_TEST_DS_PATH)
        _save_dataset(private_test_ds, PRIVATE_TEST_DS_PATH)
        
    return train_valid_ds, public_test_ds, private_test_ds


def load_subgraphed_datasets(neighbourhood_size, save=True):
    if os.path.isdir(os.path.join(SUBGRAPHED_DATASETS_DIR, f'subgraphed_{neighbourhood_size}_train_valid_ds')):
        subgraphed_train_valid_ds = _load_dataset(
            (os.path.join(SUBGRAPHED_DATASETS_DIR, f'subgraphed_{neighbourhood_size}_train_valid_ds')),
            _get_subgraphed_train_ds_spec(neighbourhood_size))
        subgraphed_public_test_ds = _load_dataset(
            (os.path.join(SUBGRAPHED_DATASETS_DIR, f'subgraphed_{neighbourhood_size}_public_test_ds')),
            _get_subgraphed_test_ds_spec(neighbourhood_size))
        subgraphed_private_test_ds = _load_dataset(
            (os.path.join(SUBGRAPHED_DATASETS_DIR, f'subgraphed_{neighbourhood_size}_private_test_ds')),
            _get_subgraphed_test_ds_spec(neighbourhood_size))

        return subgraphed_train_valid_ds, subgraphed_public_test_ds, subgraphed_private_test_ds
    
    train_valid_ds, public_test_ds, private_test_ds = load_base_datasets()

    subgraphed_train_valid_ds = _subgraph_dataset(train_valid_ds, neighbourhood_size)
    subgraphed_public_test_ds = _subgraph_dataset(public_test_ds, neighbourhood_size)
    subgraphed_private_test_ds = _subgraph_dataset(private_test_ds, neighbourhood_size)

    if save:
        _save_dataset(subgraphed_train_valid_ds,
                      os.path.join(SUBGRAPHED_DATASETS_DIR, f'subgraphed_{neighbourhood_size}_train_valid_ds'))
        _save_dataset(subgraphed_public_test_ds,
                      os.path.join(SUBGRAPHED_DATASETS_DIR, f'subgraphed_{neighbourhood_size}_public_test_ds'))
        _save_dataset(subgraphed_private_test_ds,
                      os.path.join(SUBGRAPHED_DATASETS_DIR, f'subgraphed_{neighbourhood_size}_private_test_ds'))

    return subgraphed_train_valid_ds, subgraphed_public_test_ds, subgraphed_private_test_ds
