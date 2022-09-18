import keras_tuner
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
import keras.backend as K
from keras import layers
from util import h_score_loss
from keras import callbacks
from keras.utils import np_utils
import keras_tuner as kt
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler


def custom_loss():
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return K.binary_crossentropy(y_true, y_pred)  # TODO + h_score_loss()

    # Return a function
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
            units=hp.Int('units', min_value=16, max_value=256, step=16),
            activation='relu',
            name='dense_1')
    )
    # Layer 2
    model.add(
        layers.Dense(units=hp.Int('units', min_value=16, max_value=256, step=16),
                     activation='relu',
                     name='dense_2')
    )
    # Output layer
    model.add(layers.Dense(2, activation="softmax", name='output'))
    # Tune the learning rate for the optimizer (0.01, 0.001, or 0.0001)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy', 'recall'])
    return model


def deep_model_builder(hp):
    """
    Args:
    hp - Keras tuner object
    """
    # Initialize the Sequential API and start stacking the layers
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(18,)))
    model.add(layers.Normalization(axis=None))
    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 16-256
    hp_units = hp.Int('units', min_value=16, max_value=256, step=16)
    model.add(keras.layers.Dense(units=hp_units, activation='relu', name='dense_1'))

    hp_units = hp.Int('units', min_value=16, max_value=512, step=16)
    model.add(keras.layers.Dense(units=hp_units, activation='relu', name='dense_2'))

    hp_units = hp.Int('units', min_value=16, max_value=512, step=16)
    model.add(keras.layers.Dense(units=hp_units, activation='relu', name='dense_3'))

    hp_units = hp.Int('units', min_value=16, max_value=256, step=16)
    model.add(keras.layers.Dense(units=hp_units, activation='relu', name='dense_4'))
    # Add next layers
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(2, activation="softmax", name='output'))
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy', 'recall'])
    return model


def fit_model(model, X, y, class_weight):
    callbacks_list = [
        # min_delta: Minimum change in the monitored quantity to qualify as an improvement
        # patience: Number of epochs with no improvement after which training will be stopped
        # restore_best_weights: Whether to restore model weights from the epoch with the best value of val_loss
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
    ]
    model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy(),
                                                                   keras.metrics.Recall()])
    model.fit(X, y, epochs=300, verbose=2, callbacks=callbacks_list,
              shuffle=True, validation_split=0.2, class_weight=class_weight)
    return model


def get_optimal_net(X, y, n_fold=5, seed=0, deep=False, verbose=0):
    # Instantiate the tuner
    cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    best_model = None
    best_score = 1
    for ix_train, ix_val in cv.split(X, y):
        # Get data in fold
        x_train, x_val = format_x_data(X.iloc[ix_train]), format_x_data(X.iloc[ix_val])
        y_train, y_val = format_y_data(y.iloc[ix_train]), format_y_data(y.iloc[ix_val])

        # Prepare tuner
        model_builder = deep_model_builder if deep else mlp_model_builder
        tuner = kt.RandomSearch(model_builder,  # the hyper-model
                                objective=kt.Objective('val_loss', 'min'),  # objective to optimize
                                max_trials=10,
                                executions_per_trial=5,
                                overwrite=True,
                                directory='logs',  # directory to save logs
                                project_name='xai_phishing',
                                seed=seed,
                                loss=keras.losses.BinaryCrossentropy())
        # hyper-tuning settings
        tuner.search_space_summary()

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(x_train, y_train, epochs=100, validation_data=(x_val, y_val), verbose=verbose, callbacks=[stop_early])

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
