import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyparsing import col
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
  plt.savefig("pictures/underfit.pdf", bbox_inches='tight')


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
    df = df.drop(columns=["Name"])
    # df = df.drop(df[df["Heat treatment 5 Temperature"] > 0].index)
    # df = df.drop(
    #     columns=["Heat treatment 5 Temperature", "Heat treatment 5 Time"])

    # for i in range(1, 5):
    #     col = "Heat treatment " + str(i) + " Temperature"
    #     df[col] = df[col].apply(lambda x: categorize_ht(x))

    # for i in range(1, 5):
    #     col = "Heat treatment " + str(i) + " Temperature"
    #     stage = "stage_" + str(i)
    #     df = pd.get_dummies(
    #         df, columns=[col], prefix=stage, prefix_sep='_')

    df = pd.get_dummies(df, columns=["Pressure treated"])
    df = df.drop(columns=["Pressure treated_No"])

    df = pd.get_dummies(df, columns=["Strengthening Precipitate Phase"])

    df = pd.get_dummies(df, columns=["Powder processed"])
    df = df.drop(columns=["Powder processed_No"])

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
            units=hp.Int("layer_1_units", min_value=150,
                         max_value=150, step=50),
            activation="relu"
        )
    )
    model.add(layers.Dropout(0.5))
    # model.add(
    #     layers.Dense(
    #         units=hp.Int("layer_2_units", min_value=0,
    #                      max_value=1000, step=50),
    #         activation="relu"
    #     )
    # )
    # model.add(layers.Dropout(0.5))
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
                  optimizer=tf.keras.optimizers.Nadam(0.001))
    return model


def model_builder_variable(hp, norm=0, test_layers=[1], dropout=0.2, layer_1_nodes=[50], layer_2_nodes=[50], layer_3_nodes=[50]):
    model = tf.keras.Sequential()
    model.add(norm)
    model.add(layers.Dropout(0.2))
    layer_test_space = hp.Choice("hidden_layers", values=test_layers)

    nodes = []
    nodes.append(hp.Choice("layer_1_nodes", values=layer_1_nodes))
    nodes.append(hp.Choice("layer_2_nodes", values=layer_2_nodes))
    nodes.append(hp.Choice("layer_3_nodes", values=layer_3_nodes))

    for layer_num in range(1, layer_test_space+1):
        model.add(
            layers.Dense(
                units=nodes[layer_num-1],
                activation="relu"
            )
        )
        model.add(layers.Dropout(dropout))
    # model.add(
    #     layers.Dense(
    #         units=hp.Int("layer_2_units", min_value=0,
    #                      max_value=1000, step=50),
    #         activation="relu"
    #     )
    # )
    # model.add(layers.Dropout(0.5))
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

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Nadam(0.001),
                  metrics=[tf.keras.metrics.RootMeanSquaredError(),
                           tf.keras.losses.MeanAbsoluteError()])
    return model
