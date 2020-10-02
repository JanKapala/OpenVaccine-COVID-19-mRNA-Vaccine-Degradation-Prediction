import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, TimeDistributed, Concatenate, Lambda, Bidirectional, Dropout
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD

import spektral
from spektral.layers import GraphConv, GraphAttention, GlobalAttentionPool
from data_preparation import *

TRAINING_SCORED_SEQ_LEN = 68
TESTING_SCORED_SEQ_LEN = 91
INPUT_SEQUENCE_LENGTH = None
NO_NUCLEOBASE_FEATURES = 4
NO_STRUCTURE_FEATURES = 3
NO_PREDICTED_LOOP_FEATURES = 7


def mcrmse(y_true, y_pred):
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred), axis=1))
    return tf.reduce_mean(rmse, axis=1)


def get_base_inputs():
    sequence_input = Input(shape=(INPUT_SEQUENCE_LENGTH, NO_NUCLEOBASE_FEATURES),
                           name='sequence')
    structure_input = Input(shape=(INPUT_SEQUENCE_LENGTH, NO_STRUCTURE_FEATURES),
                            name='structure')
    predicted_loop_type_input = Input(shape=(INPUT_SEQUENCE_LENGTH, NO_PREDICTED_LOOP_FEATURES),
                                      name='predicted_loop_type')

    return [sequence_input, structure_input, predicted_loop_type_input]


def get_simple_model():
    inputs = get_base_inputs()

    # Stack inputs
    stacked_inputs = Concatenate(axis=2, name='input_stacking_layer')(inputs)

    vectors_sequence = TimeDistributed(Dense(128), name='vectors_expander')(stacked_inputs)

    # Encoder
    encoder_LSTM = LSTM(256, return_state=True, name='encoder_LSTM')
    _, state_h, state_c = encoder_LSTM(vectors_sequence)

    decoder_LSTM = LSTM(256, name='decoder_LSTM', return_sequences=True)
    decoder_outputs = decoder_LSTM(vectors_sequence, initial_state=[state_h, state_c])

    # Dense layers
    x = TimeDistributed(Dense(256, activation='relu', name='dense_1'))(decoder_outputs)
    x = TimeDistributed(Dense(256, activation='relu', name='dense_2'))(x)

    # OUTPUTS
    reactivity_pred = TimeDistributed(Dense(1), name='reactivity')(x)
    deg_Mg_pH10_pred = TimeDistributed(Dense(1), name='deg_Mg_pH10')(x)
    deg_Mg_50C_pred = TimeDistributed(Dense(1), name='deg_Mg_50C')(x)

    scored_outputs = [reactivity_pred, deg_Mg_pH10_pred, deg_Mg_50C_pred]
    stacked_outputs = Concatenate(axis=2, name='stacked_outputs')(scored_outputs)

    # TESTING MODEL
    testing_model = Model(inputs=inputs, outputs={'stacked_scored_labels': stacked_outputs}, name='testing_model')

    # TRAINING MODEL
    trimmed_stacked_outputs = Lambda(lambda x: x[:, :TRAINING_SCORED_SEQ_LEN], name='trimming_layer')(stacked_outputs)
    training_model = Model(inputs=inputs, outputs={'stacked_scored_labels': trimmed_stacked_outputs},
                           name='training_model')

    training_model.compile(loss=mcrmse, optimizer='adam')
    testing_model.compile(loss=mcrmse)

    return training_model, testing_model


def get_graph_model():
    base_inputs = get_base_inputs()
    adjacency_matrix_input = Input(shape=(None, None), name='adjacency_matrix')

    inputs = base_inputs + [adjacency_matrix_input]

    stacked_base_inputs = Concatenate(axis=2, name='input_stacking_layer')(base_inputs)

    # ACTUAL MODEL
    # Stack inputs

    # Embedding
    vectors_sequence = TimeDistributed(Dense(128), name='vectors_expander')(stacked_base_inputs)

    # Graph block
    graph_layer_1 = GraphAttention(512, activation='relu')
    x = graph_layer_1([vectors_sequence, adjacency_matrix_input])

    graph_layer_2 = GraphAttention(512, activation='relu')
    x = graph_layer_2([x, adjacency_matrix_input])

    graph_layer_3 = GraphAttention(256, activation='relu')
    x = graph_layer_3([x, adjacency_matrix_input])

    graph_layer_4 = GraphAttention(256, activation='relu')
    graph_output = graph_layer_4([x, adjacency_matrix_input])

    # Encoder decoder block
    # Encoder
    encoder_LSTM = LSTM(256, return_state=True, name='encoder_LSTM')
    _, state_h, state_c = encoder_LSTM(vectors_sequence)

    # Decoder
    decoder_LSTM = LSTM(256, name='decoder_LSTM', return_sequences=True)
    decoder_outputs = decoder_LSTM(vectors_sequence, initial_state=[state_h, state_c])

    # Dense layers
    x = TimeDistributed(Dense(256, activation='relu', name='dense_1'))(decoder_outputs)
    enc_dec_output = TimeDistributed(Dense(256, activation='relu', name='dense_2'))(x)

    # Branch merging
    concat_outputs = Concatenate(axis=2, name='branch_merger')([graph_output, enc_dec_output])

    reactivity_pred = TimeDistributed(Dense(1), name='reactivity')(concat_outputs)
    deg_Mg_pH10_pred = TimeDistributed(Dense(1), name='deg_Mg_pH10')(concat_outputs)
    deg_Mg_50C_pred = TimeDistributed(Dense(1), name='deg_Mg_50C')(concat_outputs)

    # Outputs
    scored_outputs = [reactivity_pred, deg_Mg_pH10_pred, deg_Mg_50C_pred]
    stacked_outputs = Concatenate(axis=2, name='stacked_outputs')(scored_outputs)

    # TESTING MODEL
    testing_model = Model(inputs=inputs, outputs={'stacked_scored_labels': stacked_outputs}, name='testing_model')

    # TRAINING MODEL
    trimmed_stacked_outputs = Lambda(lambda x: x[:, :TRAINING_SCORED_SEQ_LEN], name='trimming_layer')(stacked_outputs)
    training_model = Model(inputs=inputs, outputs={'stacked_scored_labels': trimmed_stacked_outputs},
                           name='training_model')

    return training_model, testing_model


def get_predictions(ds, model, prediction_batch_size=64):
    prediction_ds = ds.batch(prediction_batch_size)
    predictions = model.predict(prediction_ds)['stacked_scored_labels']
    return predictions


if __name__ == '__main__':
    train_valid_ds, public_test_ds, private_test_ds = load_base_datasets()
    train_ds, valid_ds = split_into_train_and_valid(train_valid_ds, split_factor=0.3)

    training_model, testing_model = get_simple_model()

    training_model.compile(loss=mcrmse, optimizer='adam')
    testing_model.compile(loss=mcrmse)

    training_model.fit(train_ds.batch(128),
                       validation_data=valid_ds.batch(128),
                       epochs=10000,
                       initial_epoch=1001,
                       verbose=1,
                       shuffle=True)

    training_model.evaluate(valid_ds.batch(64))
    # DOESNT WORK FOR NOW - TRIM TAKES 3 ARGUMENTS LOL XD
    # testing_model.evaluate(valid_ds.map(trim).batch(64))



