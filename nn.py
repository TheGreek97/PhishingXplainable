import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam
from keras import callbacks
from keras.utils import np_utils
import keras_tuner as kt

from sklearn.preprocessing import MinMaxScaler


def get_data(x_train, y_train, x_test, y_test):
    scaler = MinMaxScaler()

    x_train_nn = scaler.fit_transform(x_train)
    y_train_nn = np_utils.to_categorical(y_train, 2)
    # x_train_nn = scaler.fit_transform(x_train[0])
    # y_train_nn = np_utils.to_categorical(y_train[0], 2)

    # x_val_nn = scaler.fit_transform(x_val[0])
    # y_val_nn = np_utils.to_categorical(y_val[0], 2)

    x_test_nn = scaler.transform(x_test)
    y_test_nn = np_utils.to_categorical(y_test, 2)  # np.asarray(y_test).astype('int32')

    return x_train_nn, y_train_nn, x_test_nn, y_test_nn  # x_val_nn, y_val_nn


def mlp_model_builder(hp):
    """
    Args:
    hp - Keras tuner object
    """
    # Initialize the Sequential API and start stacking the layers
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(18,)))
    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 16-256
    hp_units = hp.Int('units', min_value=16, max_value=256, step=16)
    model.add(keras.layers.Dense(units=hp_units, activation='relu', name='dense_1'))

    hp_units = hp.Int('units', min_value=16, max_value=256, step=16)
    model.add(keras.layers.Dense(units=hp_units, activation='relu', name='dense_2'))
    # Add next layers
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
    model.add(keras.layers.Input(shape=(18,)))
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
                  metrics=['accuracy'])
    return model


def build_optimal_nn(x_train, y_train, deep=False):
    # Instantiate the tuner
    model_builder = deep_model_builder if deep else mlp_model_builder
    tuner = kt.Hyperband(model_builder,  # the hyper-model
                         objective=kt.Objective('val_loss', 'min'),  # objective to optimize
                         max_epochs=30,
                         factor=3,  # factor which you have seen above
                         directory='logs',  # directory to save logs
                         project_name='xai_phishing',
                         loss=keras.losses.BinaryCrossentropy())
    # hyper-tuning settings
    tuner.search_space_summary()
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    # Perform hyper-tuning
    tuner.search(x_train, y_train, epochs=30, validation_split=0.2, callbacks=[stop_early])
    best_hp = tuner.get_best_hyperparameters()[0]
    # Build the model with the optimal hyper-parameters
    h_model = tuner.hypermodel.build(best_hp)
    h_model.summary()
    return h_model
