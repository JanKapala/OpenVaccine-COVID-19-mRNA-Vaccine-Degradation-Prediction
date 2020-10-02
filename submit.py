import pandas as pd
from models import get_predictions
from data_preparation import *
import tensorflow as tf
import os

def get_example_id(example_index, raw_ds):
    return raw_ds.iloc[example_index, :]['id']


def predictions_to_submission(predictions, raw_ds, missing_value=0.0):
    submission_dict = {'id_seqpos': [],
                       'reactivity': [],
                       'deg_Mg_pH10': [],
                       'deg_pH10': [],
                       'deg_Mg_50C': [],
                       'deg_50C': []
                       }
    for example_index, example in enumerate(predictions):
        example_id = get_example_id(example_index, raw_ds)
        for seqpos_index, (reactivity, deg_Mg_pH10, deg_Mg_50C) in enumerate(example):
            id_seqpos = f"{example_id}_{seqpos_index}"
            deg_pH10 = missing_value
            deg_50C = missing_value

            # add values to submission_dict
            submission_dict['id_seqpos'].append(id_seqpos)
            submission_dict['reactivity'].append(reactivity)
            submission_dict['deg_Mg_pH10'].append(deg_Mg_pH10)
            submission_dict['deg_pH10'].append(deg_pH10)
            submission_dict['deg_Mg_50C'].append(deg_Mg_50C)
            submission_dict['deg_50C'].append(deg_50C)

    return pd.DataFrame(submission_dict)


def create_submission(model, datasets, raw_datasets):
    submission_parts = []
    for ds, raw_ds in zip(datasets, raw_datasets):
        ds_predictions = get_predictions(ds, model)
        submission_part = predictions_to_submission(ds_predictions, raw_ds)
        submission_parts.append(submission_part)
    submission = pd.concat(submission_parts, ignore_index=True)
    return submission


if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    SUBMISSIONS_DIR = os.path.join(SCRIPT_DIR, 'submissions')

    # GET SAVED MODEL FROM FILE
    TESTING_MODEL_PATH = None
    testing_model = tf.keras.models.load_model(TESTING_MODEL_PATH)

    _, raw_public_test_ds, raw_private_test_ds = get_raw_datasets()
    _, public_test_ds, private_test_ds = load_base_datasets()

    submission = create_submission(testing_model, [public_test_ds, private_test_ds],
                                   [raw_public_test_ds, raw_private_test_ds])

    submission_name = 'submission.csv.zip'
    submission_path = os.path.join(SUBMISSIONS_DIR, submission_name)
    submission.to_csv(submission_path, header=True, index=False)
