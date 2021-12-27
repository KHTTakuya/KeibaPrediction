import os
import tempfile
from datetime import datetime

import optuna.integration.lightgbm as lgb
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from Keiba.dataprocess import KeibaProcessing

"""
task:
datetimeの変数化をすすめる。__init__に記載する。
"""


class KeibaPrediction:

    def __init__(self, data):
        """
        :param data: df(pandas:dataframe)
        モデルに導入する用のdataframeを引数にいれること。特にLightGBMとtensorflowは違うdataframeを使用するため注意されたし
        """
        self.data = data

    def gbm_params_keiba(self):
        """
        :return: df(pandas:dataframe)
        2着以内の確率が返ってくる。
        raceid, prediction(確率)が記載された状態。
        """
        df = self.data

        df['days'] = pd.to_datetime(df['days'])
        df = df.dropna(how='any')

        df_pred = df[df['days'] >= datetime(2021, 11, 20)]
        df_pred_droped = df_pred.drop(['flag', 'days', 'horsename', 'raceid', 'odds', 'pop'], axis=1)

        df = df[df['days'] < datetime(2021, 11, 20)]

        train_x = df.drop(['flag', 'days', 'horsename', 'raceid', 'odds', 'pop'], axis=1)
        train_y = df['flag']

        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y,
                                                            stratify=train_y,
                                                            random_state=0, test_size=0.3, shuffle=True)
        cat_cols = ['place', 'class', 'turf', 'distance', 'weather', 'condition', 'sex', 'father', 'mother',
                    'fathertype', 'fathermon', 'legtype', 'jocky', 'trainer', 'father_legtype']

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

        predict_proba = model.predict(df_pred_droped, num_iteration=model.best_iteration)

        predict = pd.DataFrame({"raceid": df_pred['raceid'],
                                "gbm_pred": predict_proba})

        return predict

    @staticmethod
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        """
        :param dataframe:
        :param shuffle:
        :param batch_size:
        :return: ds
        原則、クラス内呼び出しのみ。
        外部からの呼び出しは不可。(デバックをする場合は除く)
        """
        dataframe = dataframe.copy()
        labels = dataframe.pop('flag')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    def tensorflow_models(self, feature_layer):
        """

        :param feature_layer: df_to_tfdataからの返り値をいれる。
        :return: df(pandas:dataframe)
        2着以内の確率が返ってくる。
        raceid, prediction(確率)が記載された状態。
        """
        df = self.data

        df['days'] = pd.to_datetime(df['days'])
        drop_list = ['place', 'class', 'turf', 'distance', 'weather', 'condition', 'sex',
                     'father', 'mother', 'fathermon', 'fathertype', 'legtype', 'jocky',
                     'trainer', 'father_legtype']
        df = df.dropna(how='any')

        df = df.drop(drop_list, axis=1)

        df_pred = df[df['days'] >= datetime(2021, 11, 20)]
        df_pred_droped = df_pred.drop(['flag', 'days', 'horsename', 'raceid', 'odds', 'pop'], axis=1)

        df = df[df['days'] < datetime(2021, 11, 20)]
        df = df.drop(['days', 'horsename', 'raceid', 'odds', 'pop'], axis=1)

        train, test = train_test_split(df, test_size=0.2)
        train, val = train_test_split(train, test_size=0.2)

        batch_size = 32
        train_ds = self.df_to_dataset(train, batch_size=batch_size)
        val_ds = self.df_to_dataset(val, shuffle=False, batch_size=batch_size)
        test_ds = self.df_to_dataset(test, shuffle=False, batch_size=batch_size)

        pred_ds = tf.data.Dataset.from_tensor_slices(dict(df_pred_droped))
        pred_ds = pred_ds.batch(batch_size=batch_size)

        model = tf.keras.Sequential([
            feature_layer,
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.1),
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.1),
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit(train_ds,
                  validation_data=val_ds,
                  epochs=5)

        # loss, accuracy = model.evaluate(test_ds)

        predictions = model.predict(pred_ds)
        predict = [i for i in predictions]

        d = {
            "raceid": df_pred['raceid'],
            "tf_pred": predict
        }

        predict = pd.DataFrame(data=d)

        return predict

    def model_concatenation(self, gbm_model=None, tf_model=None):
        """
        :param gbm_model:　df(pandas:dataframe) gbm_params_keibaから持ってくる。
        :param tf_model:　df(pandas:dataframe) tensorflow_modelsから持ってくる。
        :return: df(pandas:dataframe)
        予想印やフラグの作成。
        なるべくdf.to_csvでcsvデータとして返すのがよろし
        """

        main_df = self.data
        main_df['days'] = pd.to_datetime(main_df['days'])
        main_df = main_df.dropna(how='any')

        df_pred = main_df[main_df['days'] >= datetime(2021, 11, 20)]

        df = pd.merge(gbm_model, tf_model, on='raceid', how='left')

        df = df.dropna(how='any')

        # ここは何故かエラーが起きているので一時的に止めている。もし"[]"がついている場合はコメントアウトを外して実行すること。
        # for i in range(len(df['tf_pred'])):
        #     df['tf_pred'][i] = df['tf_pred'][i].replace('[', '')
        #     df['tf_pred'][i] = df['tf_pred'][i].replace(']', '')

        df['gbm_pred'] = df['gbm_pred'].astype(float)
        df['tf_pred'] = df['tf_pred'].astype(float)

        """
        todo:
        馬番ごとでかうんとしていく。
        """
        # gbm_pred, tf_pred, logi_pred
        df['new_mark_flag'] = '×'
        df['new_flag'] = 0

        # 0.5が2個以上のフラグ作成。〇
        df['new_mark_flag'].mask((df['gbm_pred'] >= 0.5) | (df['tf_pred'] >= 0.5), '〇', inplace=True)

        # 0-1フラグの作成と指標作成(見やすくするため)
        df['new_flag'].mask(((df['gbm_pred'] * 0.45) + (df['tf_pred'] * 0.55)) >= 0.5, 1, inplace=True)
        df['malt_index'] = ((df['gbm_pred'] * 0.45) + (df['tf_pred'] * 0.55)) * 100

        df = pd.merge(df_pred, df, on='raceid', how='left')

        return df


class PredictModelTF:

    def __init__(self, data):
        raw_data = pd.read_csv(data)
        set_data = KeibaProcessing(data)
        df = set_data.data_feature_and_formating(raw_data, gbmflag=False)
        df['days'] = pd.to_datetime(df['days'])
        df = df.dropna(how='any')

        # df_drop_list = df.loc[:, 'place_odds': "distance_3ftime"]
        # start_drop_list = [x for x in df_drop_list.columns]
        # df = df.drop(start_drop_list, axis=1)

        df_pred = df[df['days'] >= datetime(2021, 11, 20)]
        df_pred_droped = df_pred.drop(['flag', 'days', 'horsename', 'odds', 'pop'], axis=1)

        df = df[df['days'] < datetime(2021, 11, 20)]
        df = df.drop(['days', 'horsename', 'raceid', 'odds', 'pop'], axis=1)

        drop_list = ['place', 'class', 'turf', 'distance', 'weather', 'condition', 'sex',
                     'father', 'mother', 'fathermon', 'fathertype', 'legtype', 'jocky',
                     'trainer', 'father_legtype']

        df_pred_droped = df_pred_droped.drop(drop_list, axis=1)
        df = df.drop(drop_list, axis=1)

        self.df = df
        self.pred_df = df_pred_droped

    def models(self):
        df = self.df
        pred_df = self.pred_df
        check_df = self.pred_df
        pred_df = pred_df.drop('raceid', axis=1)
        cleaned_df = df.dropna(how='any')

        neg, pos = np.bincount(cleaned_df['flag'])
        total = neg + pos

        EPOCHS = 100
        BATCH_SIZE = 2048
        BUFFER_SIZE = 100000

        train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
        train_df, val_df = train_test_split(train_df, test_size=0.2)

        train_labels = np.array(train_df.pop('flag'))
        bool_train_labels = train_labels != 0
        val_labels = np.array(val_df.pop('flag'))
        pred_labels = np.array(pred_df)

        train_features = np.array(train_df)
        val_features = np.array(val_df)
        pred_features = np.array(pred_df)

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
            keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
        ]

        def make_ds(features, labels):
            ds = tf.data.Dataset.from_tensor_slices((features, labels))  # .cache()
            ds = ds.shuffle(BUFFER_SIZE).repeat()
            return ds

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

        pos_ds = make_ds(pos_features, pos_labels)
        neg_ds = make_ds(neg_features, neg_labels)
        resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
        resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)
        resampled_steps_per_epoch = np.ceil(2.0 * neg / BATCH_SIZE)

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

        resampled_model = make_model()
        resampled_model.load_weights(initial_weights)

        output_layer = resampled_model.layers[-1]
        output_layer.bias.assign([0])

        val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()
        val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)

        resampled_model.fit(
            resampled_ds,
            # These are not real epochs
            steps_per_epoch=200,
            epochs=10 * EPOCHS,
            callbacks=[early_stopping],
            class_weight=class_weight,
            validation_data=val_ds)

        predictions = resampled_model.predict(pred_features)
        predict = [float(i) * 1000000 for i in predictions]

        d = {
            "raceid": check_df['raceid'],
            "tf_pred": predict
        }

        predict_df = pd.DataFrame(data=d)

        return predict_df.to_csv('test.csv', encoding='utf_8_sig')