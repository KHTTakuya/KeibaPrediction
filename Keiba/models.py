import tensorflow as tf
from imblearn.over_sampling import SMOTE
from tensorflow import keras

import optuna.integration.lightgbm as lgb
import os
import tempfile

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from Keiba.datafile import datalist


class PredictionModelGBM:

    def __init__(self, dataframe):
        df = dataframe

        df = df.astype({'distance': 'string'})

        cat_cols = ['place', 'class', 'turf', 'distance', 'weather', 'condition', 'sex', 'father', 'mother',
                    'fathertype', 'fathermon', 'legtype', 'jocky', 'trainer', 'father_legtype', 'pre_place',
                    'pre_turf', 'pre_distance']

        for c in cat_cols:
            le = LabelEncoder()
            le.fit(df[c])
            df[c] = le.transform(df[c])

        df['days'] = pd.to_datetime(df['days'])
        df = df.dropna(how='any')

        df_pred = df[df['result'] == 999]
        df_pred_drop = df_pred.drop(['pop', 'flag', 'days', 'horsename', 'raceid', 'result', 'fathertype_legtype'], axis=1)

        df = df[df['result'] != 999]
        df = df.drop(['pop', 'days', 'horsename', 'raceid', 'result', 'fathertype_legtype'], axis=1)

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

        cat_cols = ['place', 'class', 'turf', 'distance', 'weather', 'condition', 'sex', 'father', 'mother',
                    'fathertype', 'fathermon', 'legtype', 'jocky', 'trainer']

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
            num_boost_round=100,
            early_stopping_rounds=20,
        )
        best_params = model.params

        model = lgb.train(
            best_params,
            lgb_train,
            categorical_feature=cat_cols,
            valid_sets=lgb_eval,
            num_boost_round=100,  # 100
            early_stopping_rounds=20,  # 20
        )

        predict_proba = model.predict(df_pred_drop, num_iteration=model.best_iteration)

        predict = pd.DataFrame({"raceid": df_pred['raceid'],
                                "gbm_pred": predict_proba})

        return predict


"""
memo:
確率が大幅に下がる要因
・各馬の平均順位
・前走の成績での距離
"""


class PredictModelTF:

    def __init__(self, dataframe):
        df = dataframe

        df['flag_konkan'] = (df['distance'] % 400 == 0).astype(int)

        df = df.replace({'distance': [1000, 1200, 1400, 1500]}, '短距離')
        df = df.replace({'distance': [1600, 1700, 1800]}, 'マイル')
        df = df.replace({'distance': [2000, 2200, 2300, 2400]}, '中距離')
        df = df.replace({'distance': [2500, 2600, 3000, 3200, 3400, 3600]}, '長距離')

        df['flag_pre_konkan'] = (df['pre_distance'] % 400 == 0).astype(int)

        colunms_list = ['place', 'class', 'turf', 'weather', 'distance',
                        'condition', 'sex', 'pre_place', 'pre_turf']
        df = pd.get_dummies(df, columns=colunms_list)
        df = df.drop(['pop', 'father', 'mother', 'fathermon', 'fathertype', 'legtype', 'jocky',
                      'trainer', 'father_legtype', 'fathertype_legtype', 'pre_distance'], axis=1)

        previous_list = datalist.re_rename_list
        after_list = datalist.rename_list
        for i in range(len(previous_list)):
            df = df.rename(columns={previous_list[i]: after_list[i]})

        num_data = datalist.new_num_data

        num_data.remove('horsenum')

        scaler = StandardScaler()
        sc = scaler.fit(df[num_data])

        scalered_df = pd.DataFrame(sc.transform(df[num_data]), columns=num_data, index=df.index)
        df.update(scalered_df)

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
                    128, activation='relu',
                    input_shape=(train_features.shape[-1],)),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(
                    128, activation='relu',
                    input_shape=(train_features.shape[-1],)),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(
                    128, activation='relu',
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


class MergeModelDataToCsv:

    def __init__(self, main_data, gbm_data, tf_data):
        df = main_data
        df = df[df['result'] == 999]
        df = df[['raceid', 'place', 'class', 'distance', 'horsename', 'jocky', 'speedindex', 'speedmean']]

        self.df = df
        self.gbm_data = gbm_data
        self.tf_data = tf_data

    def merged_data(self):
        df = self.df

        df = pd.merge(df, self.gbm_data, on='raceid', how='left')
        df = pd.merge(df, self.tf_data, on='raceid', how='left')

        df['tf_pred'] = df['tf_pred'].astype(float)

        # gbm_pred, tf_pred, logi_pred
        df['new_mark_flag'] = '×'
        df['new_flag'] = 0

        # # 0.5が2個以上のフラグ作成。〇
        df['new_mark_flag'].mask((df['gbm_pred'] >= 0.5) | (df['tf_pred'] >= 0.5), '〇', inplace=True)

        # 0.5が3個以上のフラグ作成。◎
        df['new_mark_flag'].mask((df['gbm_pred'] >= 0.5) & (df['tf_pred'] >= 0.5), '◎', inplace=True)

        df['new_flag'].mask(((df['gbm_pred'] * 0.45) + (df['tf_pred'] * 0.55)) >= 0.5, 1, inplace=True)

        return df.to_csv('ans.csv', encoding='utf_8_sig')
