import keras_tuner
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
import keras.backend as K
from keras import layers
import pandas as pd

import nn
from util import h_score_loss
from keras import callbacks
from keras.utils import np_utils
import keras_tuner as kt
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler


INPUT_SIZE = 63


def custom_loss():
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return K.binary_crossentropy(y_true, y_pred)
    return loss


def format_x_data(x):
    scaler = MinMaxScaler()
    x_nn = scaler.fit_transform(x)
    return x_nn


def format_y_data(y):
    y_nn = np_utils.to_categorical(y, 2)
    return y_nn

"""
def mlp_model_builder(hp):
    # Initialize the Sequential API and start stacking the layers
    model = keras.Sequential()
    # Input layer
    model.add(layers.Input(shape=(INPUT_SIZE,)))
    model.add(layers.Normalization(axis=None))
    # Layer 1
    model.add(
        layers.Dense(
            units=hp.Int('units', min_value=32, max_value=64, step=16),
            activation='relu',
            name='dense_1')
    )
    model.add(
        layers.Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.25))
    )
    # Output layer
    model.add(layers.Dense(2, activation="softmax", name='output'))
    # Tune the learning rate for the optimizer (0.01, 0.001, or 0.0001)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model
    
    
def deep_model_builder(hp):
    # Initialize the Sequential API and start stacking the layers
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(INPUT_SIZE,)))
    model.add(layers.Normalization(axis=None))
    # Tune the number of units in the first Dense layer
    for i in range(2):  # hp.Int('num_layers'), 2, 4):
        model.add(keras.layers.Dense(units=hp.Int('units', min_value=128, max_value=512, step=128),
                                     activation='relu',
                                     name='dense_'+str(i)))
        model.add(keras.layers.Dropout(rate=hp.Float('dropout_' + str(i),
                                                     min_value=0.1,
                                                     max_value=0.5,
                                                     step=0.2)))
    model.add(keras.layers.Dense(2, activation="softmax", name='output'))
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model
"""


def mlp_model_builder(hp):
    """
    Args:
    hp - Keras tuner object
    """
    # Initialize the Sequential API and start stacking the layers
    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.Input(shape=(INPUT_SIZE,)))
    # Layer 1
    hp_units = hp.Int('units', min_value=16, max_value=256, step=16)  # Choose an optimal value between 16-256
    model.add(keras.layers.Dense(units=hp_units, activation='relu', name='dense_1'))
    model.add(keras.layers.Dropout(rate=hp.Float('dropout_1',
                                                 min_value=0.1,
                                                 max_value=0.5,
                                                 step=0.2)))
    # Layer 2
    hp_units = hp.Int('units', min_value=16, max_value=256, step=16)
    model.add(keras.layers.Dense(units=hp_units, activation='relu', name='dense_2'))
    model.add(keras.layers.Dropout(rate=hp.Float('dropout_2',
                                                 min_value=0.1,
                                                 max_value=0.5,
                                                 step=0.2)))
    # Output Layer
    model.add(keras.layers.Dense(2, activation="softmax", name='output'))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


def deep_model_builder(hp):
    """
    Args:
    hp - Keras tuner object
    """
    # Initialize the Sequential API and start stacking the layers
    model = keras.Sequential()
    # Input layer
    model.add(keras.layers.Input(shape=(INPUT_SIZE,)))
    # Add 4 intermediate layers
    for i in range(1, 4):
        hp_units = hp.Int('units', min_value=16, max_value=256, step=16)  # Choose an optimal value between 16-256
        model.add(keras.layers.Dense(units=hp_units, activation='relu', name='dense_' + str(i)))
        model.add(keras.layers.Dropout(rate=hp.Float('dropout_' + str(i),
                                                     min_value=0.1,
                                                     max_value=0.5,
                                                     step=0.2)))
    # Output Layer
    model.add(keras.layers.Dense(2, activation="softmax", name='output'))
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


def fit_model(model, X, y, class_weight):
    callbacks_list = [
        # min_delta: Minimum change in the monitored quantity to qualify as an improvement
        # patience: Number of epochs with no improvement after which training will be stopped
        # restore_best_weights: Whether to restore model weights from the epoch with the best value of val_loss
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10)
    ]
    model.fit(X, y, epochs=300, verbose=2, callbacks=callbacks_list,
              shuffle=True, validation_split=0.2, class_weight=class_weight)
    return model


def get_optimal_net(X, y, n_fold=5, seed=0, deep=False, verbose=0):
    # Instantiate the tuner
    nn.INPUT_SIZE = len(X.columns)  # set the input size = number of features
    cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    best_model = None
    best_score = 1
    split_ixs = cv.split(X, y)
    i = 0
    for ix_train, ix_val in split_ixs:
        # Get data in fold
        x_train, x_val = format_x_data(X.iloc[ix_train]), format_x_data(X.iloc[ix_val])
        y_train, y_val = format_y_data(y.iloc[ix_train]), format_y_data(y.iloc[ix_val])
        folder_name = "dnn_tuning" if deep else "mlp_tuning"
        folder_name += "_" + str(i)
        i += 1
        # Prepare tuner
        model_builder = deep_model_builder if deep else mlp_model_builder
        tuner = kt.RandomSearch(model_builder,  # the hyper-model
                                objective=kt.Objective('val_loss', 'min'),  # objective to optimize
                                max_trials=10,
                                executions_per_trial=5,
                                directory='logs',  # directory to save logs
                                project_name=folder_name,
                                seed=seed,
                                loss=keras.losses.BinaryCrossentropy())
        # hyper-tuning settings
        tuner.search_space_summary()

        # stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(x_train, y_train, epochs=100, validation_data=(x_val, y_val), verbose=verbose)

        h_model = tuner.get_best_models(num_models=1)[0]
        h_model.build()
        if verbose > 0:
            h_model.summary()
            tuner.results_summary()
        score = h_model.evaluate(x_val, y_val)

        # Feature importance
        #importance = permutation_importance(h_model, x_val, y_val, scoring='neg_mean_squared_error').importances_mean
        #h_score_l = h_score_loss(importance, 0.5)
        #score = score + h_score_l  # sum the two losses

        # Build the best model with the optimal hyperparameters
        # best_hps = tuner.get_best_hyperparameters()[0]
        # h_model = model_builder(best_hps)
        if score < best_score:  # minimize val loss
            best_model = h_model
            best_score = score
            print(f"Best score: {best_score}")

    best_model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy(),
                                                                        keras.metrics.Recall()])
    return best_model


def build_optimal_nn(x_train, x_val, y_train, y_val, seed=42, deep=False) -> tuple[keras.Model, dict[int, int]]:
    # Instantiate the tuner
    model_builder = deep_model_builder if deep else mlp_model_builder
    tuner = kt.Hyperband(model_builder,  # the hyper-model
                         objective=kt.Objective('val_loss', 'min'),  # objective to optimize
                         max_epochs=30,
                         factor=3,  # factor which you have seen above
                         directory='logs',  # directory to save logs
                         project_name='xai_phishing',
                         seed=seed,
                         loss=keras.losses.BinaryCrossentropy())
    # hyper-tuning settings
    tuner.search_space_summary()
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    # Perform hyper-tuning
    """
    best_w = 1
    best_model = None
    best_score = 0
    n_folds = 5
    
    y_train_v = [np_utils.to_categorical(y, 2) for y in y_train_v]
    y_val = [np_utils.to_categorical(y, 2) for y in y_val]
    k = 0
    for w in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]:
        print("Class weight: ", w)
        scores = [0, 0, 0, 0, 0]
        for k in range(0, n_folds):
    """
    w = 1
    x_val = pd.DataFrame(x_val)
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    tuner.search(x_train, y_train, validation_data=[x_val, y_val],
                 epochs=30, class_weight={0: 1, 1: w}, callbacks=[stop_early])
    best_hp = tuner.get_best_hyperparameters()[0]
    # Build the model with the optimal hyperparameters
    h_model = tuner.hypermodel.build(best_hp)
    h_model.compile(loss='binary_crossentropy')

    #     x = pd.concat([x_train, x_val])  # merge the training and the validation sets
    #     y = pd.concat([y_train, y_val])  # merge the training and the validation sets
    #     h_model.fit(x, y)
    """
    scores[k] = h_model.evaluate(x_v, y_v, verbose=0)
    avg_score = sum(scores) / n_folds
    # print(avg_score, "best: ", best_score)
    if avg_score > best_score:
        best_model = h_model
        best_w = w
        best_score = avg_score
    best_model.summary()
    return best_model, {0: 1, 1: best_w}
    """
    return h_model, {0: 1, 1: w}

