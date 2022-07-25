import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential

import keras_tuner as kt


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  # plt.ylim([0, 1200])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)


def normalize(train_features):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    first = np.array(train_features[:1])

    with np.printoptions(precision=2, suppress=True):
        print('First example:', first)
        print()
        print('Normalized:', normalizer(first).numpy())

    return normalizer


def model_builder(hp, norm=0):
    model = tf.keras.Sequential()
    model.add(norm)
    model.add(
        layers.Dense(
            units=hp.Int("layer_1_units", min_value=1, max_value=50, step=1),
            activation="relu"
        )
    )
    model.add(layers.Dropout(0.2))

    second_layer_units = hp.Int(
        "layer_2_units", min_value=1, max_value=50, step=1)

    if True:
        model.add(
            layers.Dense(
                units=second_layer_units,
                activation="relu"
            )
        )
        model.add(layers.Dropout(0.2))

    model.add(layers.Dense(1))

    # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # hp_optimizer = hp.Choice(
    #     'opimizer', values=["adam", "nadam"])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Nadam(0.01))
    return model
