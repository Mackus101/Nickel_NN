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


def categorize_ht(val):
    if val >= 1000:
        return "Anneal"
    elif val >= 750:
        return "Hot"
    elif val > 1:
        return "Warm"
    else:
        return "No_HT"


def clean_dataframe(df, label):
    df = df.drop(columns="Name")
    df = df.drop(df[df["Heat treatment 5 Temperature"] > 0].index)
    df = df.drop(
        columns=["Heat treatment 5 Temperature", "Heat treatment 5 Time"])

    for i in range(1, 5):
        col = "Heat treatment " + str(i) + " Temperature"
        df[col] = df[col].apply(lambda x: categorize_ht(x))

    for i in range(1, 5):
        col = "Heat treatment " + str(i) + " Temperature"
        stage = "stage_" + str(i)
        df = pd.get_dummies(
            df, columns=[col], prefix=stage, prefix_sep='_')

    unused_labels = ['Tensile Strength, Yield',
                     'Elongation at Break', "Tensile Strength, Ultimate"]
    unused_labels.remove(label)

    df = df.dropna(subset=label)
    df = df.drop(columns=unused_labels)
    df = df.fillna(0)
    df = df.astype("float32")
    return df


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
    model.add(layers.Dropout(0.2))
    model.add(
        layers.Dense(
            units=hp.Int("layer_1_units", min_value=200,
                         max_value=200, step=1),
            activation="relu"
        )
    )
    model.add(layers.Dropout(0.5))
    model.add(
        layers.Dense(
            units=hp.Int("layer_2_units", min_value=200,
                         max_value=200, step=1),
            activation="relu"
        )
    )
    model.add(layers.Dropout(0.5))
    # model.add(
    #     layers.Dense(
    #         units=hp.Int("layer_3_units", min_value=1, max_value=100, step=1),
    #         activation="relu"
    #     )
    # )
    # model.add(layers.Dropout(0.2))
    # model.add(
    #     layers.Dense(
    #         units=hp.Int("layer_4_units", min_value=1, max_value=100, step=1),
    #         activation="relu"
    #     )
    # )
    # model.add(layers.Dropout(0.2))
    # model.add(layers.Dropout(0.2))
    # model.add(
    #     layers.Dense(
    #         units=hp.Int("layer_5_units", min_value=1, max_value=100, step=1),
    #         activation="relu"
    #     )
    # )
    # model.add(layers.Dropout(0.2))

    model.add(layers.Dense(1))

    # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # hp_optimizer = hp.Choice(
    #     'opimizer', values=["adam", "nadam"])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Nadam(0.005))
    return model
