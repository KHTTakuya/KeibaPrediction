from datetime import datetime

import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Keiba.dataprocess import KeibaProcessing


class TestTFModel:

    def __init__(self, data):
        raw_data = pd.read_csv(data)
        set_data = KeibaProcessing(data)
        df = set_data.data_feature_and_formating(raw_data, gbmflag=False)
        df['days'] = pd.to_datetime(df['days'])
        df = df.dropna(how='any')

        df_pred = df[df['days'] >= datetime(2021, 11, 20)]
        df_pred_droped = df_pred.drop(['flag', 'days', 'horsename', 'raceid', 'odds', 'pop'], axis=1)

        df = df[df['days'] < datetime(2021, 11, 20)]
        df = df.drop(['days', 'horsename', 'raceid', 'odds', 'pop'], axis=1)

        self.df = df
        self.pred_df = df_pred_droped

    def check_df(self):
        df = self.df
        return df.to_csv('main_tst_filter.csv', encoding='utf_8-sig')

    def bitcount(self):
        raw_df = self.df
        neg, pos = np.bincount(raw_df['flag'])
        total = neg + pos
        return 'Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total)

    def models(self):
        df = self.df
        drop_list = ['place', 'class', 'turf', 'distance', 'weather', 'condition', 'sex',
                     'father', 'mother', 'fathermon', 'fathertype', 'legtype', 'jocky',
                     'trainer', 'father_legtype']
        cleaned_df = df.drop(drop_list, axis=1)
        neg, pos = np.bincount(cleaned_df['flag'])
        total = neg + pos

        train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
        train_df, val_df = train_test_split(train_df, test_size=0.2)

        train_labels = np.array(train_df.pop('flag'))
        bool_train_labels = train_labels != 0
        val_labels = np.array(val_df.pop('flag'))
        test_labels = np.array(test_df.pop('flag'))

        train_features = np.array(train_df)
        val_features = np.array(val_df)
        test_features = np.array(test_df)

        train_features = np.clip(train_features, -5, 5)
        val_features = np.clip(val_features, -5, 5)
        test_features = np.clip(test_features, -5, 5)

        pos_df = pd.DataFrame(train_features[bool_train_labels], columns=train_df.columns)
        neg_df = pd.DataFrame(train_features[~bool_train_labels], columns=train_df.columns)

        pos_features = train_features[bool_train_labels]
        neg_features = train_features[~bool_train_labels]

        pos_labels = train_labels[bool_train_labels]
        neg_labels = train_labels[~bool_train_labels]

        METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
        ]

        EPOCHS = 100
        BATCH_SIZE = 2048
        BUFFER_SIZE = 100000

        def make_ds(features, labels):
            ds = tf.data.Dataset.from_tensor_slices((features, labels))  # .cache()
            ds = ds.shuffle(BUFFER_SIZE).repeat()
            return ds

        pos_ds = make_ds(pos_features, pos_labels)
        neg_ds = make_ds(neg_features, neg_labels)
        resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
        resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)
        resampled_steps_per_epoch = np.ceil(2.0 * neg / BATCH_SIZE)

        def make_model(metrics=METRICS, output_bias=None):
            if output_bias is not None:
                output_bias = tf.keras.initializers.Constant(output_bias)
            model = keras.Sequential([
                keras.layers.Dense(
                    16, activation='relu',
                    input_shape=(train_features.shape[-1],)),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(1, activation='sigmoid',
                                   bias_initializer=output_bias),
            ])

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=metrics)

            return model

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_prc',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True)

        initial_bias = np.log([pos / neg])

        model = make_model(output_bias=initial_bias)
        results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)

        initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
        model.save_weights(initial_weights)

        def plot_metrics(history):
            metrics = ['loss', 'prc', 'precision', 'recall']
            for n, metric in enumerate(metrics):
                name = metric.replace("_", " ").capitalize()
                plt.subplot(2, 2, n + 1)
                plt.plot(history.epoch, history.history[metric], label='Train')
                plt.plot(history.epoch, history.history['val_' + metric],
                         color='forestgreen', linestyle="--", label='Val')
                plt.xlabel('Epoch')
                plt.ylabel(name)
                if metric == 'loss':
                    plt.ylim([0, plt.ylim()[1]])
                elif metric == 'auc':
                    plt.ylim([0.8, 1])
                else:
                    plt.ylim([0, 1])

                plt.legend()

                return plt.show()

        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)

        class_weight = {0: weight_for_0, 1: weight_for_1}

        resampled_model = make_model()
        resampled_model.load_weights(initial_weights)

        # Reset the bias to zero, since this dataset is balanced.
        output_layer = resampled_model.layers[-1]
        output_layer.bias.assign([0])

        val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()
        val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)

        resampled_history = resampled_model.fit(
            resampled_ds,
            # These are not real epochs
            steps_per_epoch=20,
            epochs=10 * EPOCHS,
            callbacks=[early_stopping],
            validation_data=(val_ds))

        return plot_metrics(resampled_history)