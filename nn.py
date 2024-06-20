import keras_tuner
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
import keras.backend as K
from keras import layers

import nn
from util import h_score_loss
from keras import callbacks
from keras.utils import np_utils
import keras_tuner as kt
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler


INPUT_SIZE = 18


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


def mlp_model_builder(hp):
    """
    Args:
    hp - Keras tuner object
    """
    # Initialize the Sequential API and start stacking the layers
    model = keras.Sequential()
    # Input layer
    model.add(layers.Input(shape=(18,)))
    model.add(layers.Normalization(axis=None))
    # Layer 1
    model.add(
        layers.Dense(
            units=hp.Int('units', min_value=16, max_value=64, step=16),
            activation='relu',
            name='dense_1')
    )
    # Layer 2
    model.add(
        layers.Dense(units=hp.Int('units', min_value=16, max_value=64, step=16),
                     activation='relu',
                     name='dense_2')
    )
    # Output layer
    model.add(layers.Dense(2, activation="softmax", name='output'))
    # Tune the learning rate for the optimizer (0.01, 0.001, or 0.0001)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy', 'recall'])
    return model


def deep_model_builder(hp):
    """
    Args:
    hp - Keras tuner object
    """
    # Initialize the Sequential API and start stacking the layers
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(INPUT_SIZE,)))
    model.add(layers.Normalization(axis=None))
    # Tune the number of units in the first Dense layer

    model.add(keras.layers.Dense(units=hp.Int('units', min_value=16, max_value=64, step=16),
                                 activation='relu',
                                 name='dense_1'))

    model.add(keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=128, step=16),
                                 activation='relu',
                                 name='dense_2'))

    model.add(keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=128, step=16),
                                 activation='relu',
                                 name='dense_3'))

    model.add(keras.layers.Dense(units=hp.Int('units', min_value=16, max_value=64, step=16),
                                 activation='relu',
                                 name='dense_4'))
    # Add next layers
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(2, activation="softmax", name='output'))
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy', 'recall'])
    return model


def fit_model(model, X, y, class_weight):
    callbacks_list = [
        # min_delta: Minimum change in the monitored quantity to qualify as an improvement
        # patience: Number of epochs with no improvement after which training will be stopped
        # restore_best_weights: Whether to restore model weights from the epoch with the best value of val_loss
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10)
    ]
    model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy(),
                                                                   keras.metrics.Recall()])
    model.fit(X, y, epochs=300, verbose=2, callbacks=callbacks_list,
              shuffle=True, validation_split=0.2, class_weight=class_weight)
    return model


def get_optimal_net(X, y, n_fold=5, seed=0, deep=False, verbose=0):
    # Instantiate the tuner
    nn.INPUT_SIZE = len(X.iloc[0].columns)  # set the input size = number of features
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

        # Build the best model with the optimal hyper-parameters
        # best_hps = tuner.get_best_hyperparameters()[0]
        # h_model = model_builder(best_hps)
        if score < best_score:  # minimize val loss
            best_model = h_model
            best_score = score
            print(f"Best score: {best_score}")
    return best_model
