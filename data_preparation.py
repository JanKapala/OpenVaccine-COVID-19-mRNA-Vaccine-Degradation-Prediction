import os

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import lil_matrix

# CONSTANTS
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATASETS_DIR = os.path.join(SCRIPT_DIR, 'datasets')
SUBMISSIONS_DIR = os.path.join(SCRIPT_DIR, 'submissions')
RAW_TRAIN_DS_PATH = os.path.join(DATASETS_DIR, 'train.json')
RAW_TEST_DS_PATH = os.path.join(DATASETS_DIR, 'test.json')
SAMPLE_SUBMISSION_PATH = os.path.join(SUBMISSIONS_DIR, 'sample_submission.csv')
FEATURE_NAMES = ['sequence', 'structure', 'predicted_loop_type', 'adjacency_matrix']
ERROR_LABEL_NAMES = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C']
NORMAL_LABEL_NAMES = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
ALL_LABEL_NAMES = ERROR_LABEL_NAMES + NORMAL_LABEL_NAMES
SCORED_LABEL_NAMES = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']


# HELPER PREP FUNCTIONS
def _struct2adjmatrix(structure):
    N = len(structure)
    adjacency_matrix = lil_matrix((N, N))
    stack = []
    for i, symbol in enumerate(structure):
        if symbol == '.':
            continue
        elif symbol == '(':
            stack.append(i)
        elif symbol == ')':
            j = stack.pop()
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1
    return adjacency_matrix.tocsr()


def _add_adjacency_matrix_column(raw_ds):
    raw_ds.insert(4, 'adjacency_matrix', None)
    raw_ds['adjacency_matrix'] = raw_ds['structure'].map(lambda structure: _struct2adjmatrix(structure).toarray())


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


# class _AdjacencyMatrixEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass
#
#     def fit(self, X, y=None, *args, **kwargs):
#         return self
#
#     def transform(self, X, *args, **kwargs):
#         for title in self.titles:
#             X.insert(len(X.columns), title, 0)
#         X = X.apply(self.set_titles, axis='columns')
#         X.drop(columns=['Name'], inplace=True)
#         return X


# PUBLIC FUNCTIONS
def get_raw_datasets():
    raw_train_valid_ds = pd.read_json(RAW_TRAIN_DS_PATH, lines=True)
    raw_test_ds = pd.read_json(RAW_TEST_DS_PATH, lines=True)

    _add_adjacency_matrix_column(raw_train_valid_ds)
    _add_adjacency_matrix_column(raw_test_ds)

    raw_public_test_ds = raw_test_ds.loc[raw_test_ds['seq_length'] == 107]
    raw_private_test_ds = raw_test_ds.loc[raw_test_ds['seq_length'] == 130]

    return raw_train_valid_ds, raw_public_test_ds, raw_private_test_ds


def get_datasets():
    raw_train_valid_ds, raw_public_test_ds, raw_private_test_ds = get_raw_datasets()
    train_valid_ds = _create_train_valid_ds(raw_train_valid_ds, FEATURE_NAMES, ALL_LABEL_NAMES)
    public_test_ds = _create_test_ds(raw_public_test_ds, FEATURE_NAMES)
    private_test_ds = _create_test_ds(raw_private_test_ds, FEATURE_NAMES)

    return train_valid_ds, public_test_ds, private_test_ds

def convert_to_datasets(raw_train_valid_ds, raw_public_test_ds, raw_private_test_ds):
    train_valid_ds = _create_train_valid_ds(raw_train_valid_ds, FEATURE_NAMES, ALL_LABEL_NAMES)
    public_test_ds = _create_test_ds(raw_public_test_ds, FEATURE_NAMES)
    private_test_ds = _create_test_ds(raw_private_test_ds, FEATURE_NAMES)

    return train_valid_ds, public_test_ds, private_test_ds


def ds_summary(ds, name):
    print(f"{name}")
    print(ds)
    print(f"Cardinality: {ds.cardinality()}")
    print('\n')
    print(f"{name} example")
    print(next(iter(ds)))


def only_stacked_scored_labels(x, y):
    tensors_to_stack = [y[scored_label] for scored_label in SCORED_LABEL_NAMES]
    stacked_scored_labels = tf.stack(tensors_to_stack, axis=1)
    y = {'stacked_scored_labels': stacked_scored_labels}
    return x, y


def split_into_train_and_valid(train_valid_ds, split_factor=0.3):
    length = train_valid_ds.cardinality().numpy()
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
