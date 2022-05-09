import gc
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras

import optuna.integration.lightgbm as lgb
import os
import tempfile

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class TestPredictionModelGBM:

    def __init__(self, dataframe):
        df = dataframe

        df = df.astype({'distance': 'string', 'pre_distance': 'string'})

        cat_cols = ['place', 'turf', 'distance', 'weather', 'condition', 'sex',
                    'horsename', 'trainer', 'pre_place', 'pre_turf', 'pre_distance']

        for c in cat_cols:
            le = LabelEncoder()
            le.fit(df[c])
            df[c] = le.transform(df[c])

        df['days'] = pd.to_datetime(df['days'])
        df = df.dropna(how='any')

        drop_list = ['days', 'raceid', 'result', 'racenum', 'class', 'jocky', 'horsecount',
                     'weight', 'father', 'mother', 'fathertype', 'legtype', 'fathermon']

        df_pred = df[df['result'] == 999]
        df_pred_drop = df_pred.drop(drop_list, axis=1)
        df_pred_drop = df_pred_drop.drop('flag', axis=1)

        df = df[df['result'] != 999]
        df = df.drop(drop_list, axis=1)

        self.df = df
        self.df_pred = df_pred
        self.df_pred_drop = df_pred_drop

    def model(self):
        df = self.df
        df_pred = self.df_pred
        df_pred_drop = self.df_pred_drop

        train_x = df.drop('flag', axis=1)
        train_y = df['flag']

        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y,
                                                            stratify=train_y,
                                                            random_state=0, test_size=0.3, shuffle=True)

        cat_cols = ['place', 'turf', 'distance', 'weather', 'condition', 'sex',
                    'horsename', 'trainer', 'pre_place', 'pre_turf', 'pre_distance']
        # cat_cols = ['place', 'turf', 'distance', 'weather', 'condition', 'sex',
        #             'horsename', 'trainer']

        sm = SMOTE()
        x_resampled, y_resampled = sm.fit_resample(X_train, y_train)

        lgb_train = lgb.Dataset(x_resampled, y_resampled, categorical_feature=cat_cols)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, categorical_feature=cat_cols)

        params = {
            'task': 'predict',
            'objective': 'binary',
            'verbosity': -1,
        }

        model = lgb.train(
            params,
            lgb_train,
            categorical_feature=cat_cols,
            valid_sets=lgb_eval,
            num_boost_round=50,
            early_stopping_rounds=10,
        )
        best_params = model.params

        model = lgb.train(
            best_params,
            lgb_train,
            categorical_feature=cat_cols,
            valid_sets=lgb_eval,
            num_boost_round=50,  # 100
            early_stopping_rounds=10,  # 20
        )

        predict_proba = model.predict(df_pred_drop, num_iteration=model.best_iteration)

        predict = pd.DataFrame({"raceid": df_pred['raceid'],
                                "gbm_pred": predict_proba})

        preds = np.round(model.predict(X_test))

        print('Accuracy score = \t {}'.format(accuracy_score(y_test, preds)))
        print('Precision score = \t {}'.format(precision_score(y_test, preds)))
        print('Recall score =   \t {}'.format(recall_score(y_test, preds)))
        print('F1 score =      \t {}'.format(f1_score(y_test, preds)))

        return predict


"""
memo:
確率が大幅に下がる要因
・各馬の平均順位
・前走の成績での距離
"""


class TestPredictModelTF:

    def __init__(self, dataframe):
        df = dataframe

        df = df.replace({'distance': [1000, 1200, 1400, 1500]}, 'sprint')
        df = df.replace({'distance': [1600, 1700, 1800]}, 'mile')
        df = df.replace({'distance': [2000, 2200, 2300, 2400]}, 'middle')
        df = df.replace({'distance': [2500, 2600, 3000, 3200, 3400, 3600]}, 'stayer')

        df = df.replace({'pre_distance': [1000, 1200, 1400, 1500]}, 'sprint')
        df = df.replace({'pre_distance': [1600, 1700, 1800]}, 'mile')
        df = df.replace({'pre_distance': [2000, 2200, 2300, 2400]}, 'middle')
        df = df.replace({'pre_distance': [2500, 2600, 3000, 3200, 3400, 3600]}, 'stayer')

        columns_list = ['place', 'class', 'turf', 'weather', 'distance',
                        'condition', 'sex', 'pre_place', 'pre_turf', 'pre_distance']

        df = pd.get_dummies(df, columns=columns_list)
        df = df.drop(['father', 'mother', 'fathermon', 'fathertype', 'legtype', 'jocky', 'trainer'], axis=1)

        df['days'] = pd.to_datetime(df['days'])
        df = df.dropna(how='any')

        df_pred = df[df['result'] == 999]
        df_pred_droped = df_pred.drop(['flag', 'days', 'horsename', 'result'], axis=1)

        df = df[df['result'] != 999]
        df = df.drop(['days', 'horsename', 'raceid', 'result'], axis=1)

        self.df = df
        self.pred_df = df_pred_droped

    def models(self):
        df = self.df
        pred_df = self.pred_df
        predictions_df = pred_df.drop('raceid', axis=1)
        cleaned_df = df.copy()
        neg, pos = np.bincount(cleaned_df['flag'])
        total = neg + pos

        train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
        train_df, val_df = train_test_split(train_df, test_size=0.2)

        train_labels = np.array(train_df.pop('flag'))
        bool_train_labels = train_labels != 0
        val_labels = np.array(val_df.pop('flag'))
        pred_labels = np.array(predictions_df)

        train_features = np.array(train_df)
        val_features = np.array(val_df)
        pred_features = np.array(predictions_df)

        train_features = np.clip(train_features, -5, 5)
        val_features = np.clip(val_features, -5, 5)
        pred_features = np.clip(pred_features, -5, 5)

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
            keras.metrics.AUC(name='prc', curve='PR'),
        ]

        EPOCHS = 100
        BATCH_SIZE = 2048
        BUFFER_SIZE = 100000

        def make_ds(features, labels):
            ds = tf.data.Dataset.from_tensor_slices((features, labels))
            ds = ds.shuffle(BUFFER_SIZE).repeat()
            return ds

        pos_ds = make_ds(pos_features, pos_labels)
        neg_ds = make_ds(neg_features, neg_labels)

        def make_model(metrics=METRICS, output_bias=None):
            if output_bias is not None:
                output_bias = tf.keras.initializers.Constant(output_bias)
            model = keras.Sequential([
                keras.layers.Dense(
                    256, activation='relu',
                    input_shape=(train_features.shape[-1],)),
                keras.layers.Dense(
                    128, activation='relu',
                    input_shape=(train_features.shape[-1],)),
                keras.layers.Dense(
                    128, activation='relu',
                    input_shape=(train_features.shape[-1],)),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(
                    256, activation='relu',
                    input_shape=(train_features.shape[-1],)),
                keras.layers.Dense(
                    128, activation='relu',
                    input_shape=(train_features.shape[-1],)),
                keras.layers.Dense(
                    128, activation='relu',
                    input_shape=(train_features.shape[-1],)),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(
                    256, activation='relu',
                    input_shape=(train_features.shape[-1],)),
                keras.layers.Dense(
                    128, activation='relu',
                    input_shape=(train_features.shape[-1],)),
                keras.layers.Dense(
                    128, activation='relu',
                    input_shape=(train_features.shape[-1],)),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(
                    256, activation='relu',
                    input_shape=(train_features.shape[-1],)),
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

        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)

        class_weight = {0: weight_for_0, 1: weight_for_1}

        weighted_model = make_model()
        weighted_model.load_weights(initial_weights)

        output_layer = weighted_model.layers[-1]
        output_layer.bias.assign([0])

        val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()
        val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)

        weighted_model.fit(
            train_features,
            train_labels,
            batch_size=BATCH_SIZE,
            steps_per_epoch=50,
            epochs=10 * EPOCHS,
            callbacks=[early_stopping],
            validation_data=(val_features, val_labels),
            class_weight=class_weight)

        predictions = weighted_model.predict(pred_features)
        predict = [float(i) for i in predictions]

        d = {
            "raceid": pred_df['raceid'],
            "tf_pred": predict
        }

        predict_df = pd.DataFrame(data=d)

        return predict_df


class TestMergeModelDataToCsv:

    def __init__(self, main_data, gbm_data, tf_data):
        df = main_data
        df = df[df['result'] == 999]
        df = df[['raceid', 'place', 'class', 'distance', 'horsename', 'jocky',
                 'racenum', 'horsecount', 'speedindex', 'last_race_index']]

        self.df = df
        self.gbm_data = gbm_data
        self.tf_data = tf_data

    def merged_data(self):
        df = self.df

        df = pd.merge(df, self.gbm_data, on='raceid', how='left')
        df = pd.merge(df, self.tf_data, on='raceid', how='left')

        df['tf_pred'] = df['tf_pred'].astype(float)

        # gbm_pred, tf_pred
        df['probability'] = ((df['gbm_pred'] * 0.55) + (df['tf_pred'] * 0.45)).round(2)

        df = df.drop_duplicates(subset=['raceid', 'horsename'])

        return df.to_csv('ans_test2.csv', encoding='utf_8_sig')
