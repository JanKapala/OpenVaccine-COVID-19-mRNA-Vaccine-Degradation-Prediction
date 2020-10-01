from IPython.display import display

import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
from matplotlib import cm
import joypy

from ipynb.draw import draw_struct
from data_preparation import NORMAL_LABEL_NAMES, ERROR_LABEL_NAMES


def raw_ds_summary(raw_ds):
    print('EXAMPLES:')
    display(raw_ds.head())
    print()
    print('SIMPLE STATISTICS:')
    display(raw_ds.describe())
    print()
    print('POSSIBLE SEQUENCES LENGTHS')
    display(raw_ds[['seq_length', 'seq_scored']].drop_duplicates())


def visualize_example_errors(example):
    plots_no = len(NORMAL_LABEL_NAMES)
    plot_width = 5
    plot_height = 5
    fig, axes = plt.subplots(1, 5, figsize=(plot_width * plots_no, plot_height))

    for i, ax in enumerate(axes.flat):
        y = example[NORMAL_LABEL_NAMES[i]]
        x = range(len(y))
        dy = example[ERROR_LABEL_NAMES[i]]
        ax.errorbar(x, y, yerr=dy, fmt='.k', ecolor='gray')


def visualize_median_errors(raw_ds):
    plots_no = len(ERROR_LABEL_NAMES)
    plot_width = 5
    plot_height = 5
    fig, axes = plt.subplots(1, 5, figsize=(plot_width * plots_no, plot_height))

    for i, ax in enumerate(axes.flat):
        column = raw_ds[ERROR_LABEL_NAMES[i]]
        x = np.array(column.values.tolist())
        medians = np.median(x, axis=0)
        ax.scatter(range(len(medians)), medians)


def _max_for_cut_index(x, cut_index, the_range):
    the_max = 0
    for i in range(the_range):
        new_max = x[i][cut_index]
        if new_max > the_max:
            the_max = new_max
    return the_max


def _find_cut_index(x, outlier_threshold):
    bases_range = x.shape[0]
    i_min = 0
    i_max = x.shape[1]
    while True:
        if i_min + 1 >= i_max:
            return i_min

        index_to_check = int((i_max + i_min) / 2)
        found_val = _max_for_cut_index(x, index_to_check, bases_range)

        if found_val > outlier_threshold:
            i_max = index_to_check
        else:
            i_min = index_to_check


def visualize_column_error_label(column, error_label_name, outliers_threshold=2):
    # Get and prepare data
    x = np.array(column.values.tolist())
    x = np.transpose(x)
    x = np.array([sorted(base) for base in x])

    # Calculate
    original_data_size = x.shape[0] * x.shape[1]
    cut_index = _find_cut_index(x, outliers_threshold)
    x = x[:, :cut_index]
    cut_data_size = x.shape[0] * x.shape[1]

    data_percentage = cut_data_size / original_data_size * 100

    df = pd.DataFrame()
    for base_index in range(x.shape[0]):
        df[base_index] = x[base_index]
    f, _ = joypy.joyplot(df, overlap=2, colormap=cm.OrRd_r, linecolor='black', linewidth=0.5, fade=True,
                         figsize=(5, 12),
                         title=f"{error_label_name} distributions for bases across all examples (without examples "
                               f"containing outlier for even one base - {data_percentage:.2f}% of data)")


def visualize_column_normal_label(column):
    x = np.array(column.values.tolist())
    df = pd.DataFrame(x)
    display(df.describe())
    fig, axes = joypy.joyplot(df, overlap=2, colormap=cm.OrRd_r, linecolor='black', linewidth=0.5, fade=True,
                              figsize=(24, 12))
    plt.show(fig)


def visualize_raw_example(example, **kwargs):
    sequence = example['sequence']
    structure = example['structure']
    seq_length = example['seq_length']
    seq_scored = example['seq_scored']
    seq_not_scored = seq_length - seq_scored
    draw_struct(sequence, structure, alpha=np.concatenate([np.ones(seq_scored), 0.3 * np.ones(seq_not_scored)]),
                **kwargs)


def visualize_random_raw_examples(raw_ds, examples_no=5, **kwargs):
    example_indices = random.sample(range(len(raw_ds)), examples_no)
    plot_size = 15
    fig, axes = plt.subplots(1, examples_no, figsize=(plot_size * examples_no, plot_size))
    for i, ax in enumerate(axes.flat):
        visualize_raw_example(raw_ds.iloc[example_indices[i], :], ax=ax, **kwargs)


def construct_column_desc(columns, name):
    if name[-1] == 's':
        name = name[:-1]
    column_name_key_name = name.lower() + '_name'
    column_desc_dict = {column_name_key_name:[], 'tensor_shape':[]}
    print(name.upper()+'S:')
    for column_name, values in columns.items():
        column_desc_dict[column_name_key_name].append(column_name)
        column_desc_dict['tensor_shape'].append(values.shape)

    display(pd.DataFrame(column_desc_dict))


def inspect_dataset_columns(ds):
    example = next(iter(ds))

    if type(example) == dict:
        features = example
        construct_column_desc(features, 'features')

    elif type(example) == tuple:
        features = example[0]
        labels = example[1]
        construct_column_desc(features, 'features')
        print()
        construct_column_desc(labels, 'labels')
    else:
        raise Exception('Invalid dataset, should be tf.data.Dataset with internal structure of example: ({...},{...})')
