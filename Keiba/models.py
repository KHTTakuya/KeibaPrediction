from datetime import datetime

import optuna.integration.lightgbm as lgb
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

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

        df_pred = df[df['days'] >= datetime(2021, 10, 24)]
        df_pred_droped = df_pred.drop(['flag', 'days', 'horsename', 'raceid', 'odds', 'pop'], axis=1)

        df = df[df['days'] < datetime(2021, 10, 24)]

        train_x = df.drop(['flag', 'days', 'horsename', 'raceid', 'odds', 'pop'], axis=1)
        train_y = df['flag']

        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y,
                                                            stratify=train_y,
                                                            random_state=0, test_size=0.3, shuffle=True)
        cat_cols = ['place', 'class', 'turf', 'distance', 'weather', 'condition', 'sex', 'father', 'mother',
                    'fathertype', 'fathermon', 'legtype', 'jocky', 'trainer', 'father_legtype']

        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
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
        df = df.dropna(how='any')

        df_pred = df[df['days'] >= datetime(2021, 10, 24)]
        df_pred_droped = df_pred.drop(['flag', 'days', 'horsename', 'raceid', 'odds', 'pop'], axis=1)

        df = df[df['days'] < datetime(2021, 10, 24)]
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
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
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

    def logistic_model(self):
        """
        :return: df(pandas:dataframe)
        2着以内の確率が返ってくる。
        raceid, prediction(確率)が記載された状態。
        """
        df = self.data

        df['days'] = pd.to_datetime(df['days'])
        df = df.dropna(how='any')

        drop_cat_cols = ['father', 'mother', 'fathermon', 'fathertype', 'legtype', 'jocky', 'trainer', 'father_legtype']
        dummies_cols = ['place', 'class', 'turf', 'weather', 'distance', 'condition', 'sex']

        df_droped = df.drop(drop_cat_cols, axis=1)
        df_droped = pd.get_dummies(df_droped, drop_first=True, columns=dummies_cols)

        df_pred = df_droped[df_droped['days'] >= datetime(2021, 10, 24)]
        df_pred_droped = df_pred.drop(['flag', 'days', 'horsename', 'raceid'], axis=1)

        df_droped = df_droped[df_droped['days'] < datetime(2021, 10, 24)]
        train_x = df_droped.drop(['flag', 'days', 'horsename', 'raceid'], axis=1)
        train_y = df_droped['flag']

        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y,
                                                            stratify=train_y,
                                                            random_state=0, test_size=0.3, shuffle=True)

        logi_model = LogisticRegression(random_state=0, C=0.5, multi_class='auto', solver='lbfgs')
        logi_model.fit(X_train, y_train)
        predictions = logi_model.predict_proba(df_pred_droped)

        predict_list = []
        for i in range(len(predictions)):
            predict_list.append(predictions[i][1])

        predict = pd.DataFrame({"raceid": df_pred['raceid'],
                                "logi_pred": predict_list})

        return predict

    def model_concatenation(self, gbm_model=None, tf_model=None, logi_model=None):
        """
        :param gbm_model:　df(pandas:dataframe) gbm_params_keibaから持ってくる。
        :param tf_model:　df(pandas:dataframe) tensorflow_modelsから持ってくる。
        :param logi_model:　df(pandas:dataframe) logistic_modelから持ってくる。
        :return: df(pandas:dataframe)
        予想印やフラグの作成。
        なるべくdf.to_csvでcsvデータとして返すのがよろし
        """

        main_df = self.data
        main_df['days'] = pd.to_datetime(main_df['days'])
        main_df = main_df.dropna(how='any')

        df_pred = main_df[main_df['days'] >= datetime(2021, 10, 24)]

        df = pd.merge(gbm_model, tf_model, on='raceid', how='left')
        df = pd.merge(df, logi_model, on='raceid', how='left')

        df = df.dropna(how='any')

        # ここは何故かエラーが起きているので一時的に止めている。もし"[]"がついている場合はコメントアウトを外して実行すること。
        # for i in range(len(df['tf_pred'])):
        #     df['tf_pred'][i] = df['tf_pred'][i].replace('[', '')
        #     df['tf_pred'][i] = df['tf_pred'][i].replace(']', '')

        df['gbm_pred'] = df['gbm_pred'].astype(float)
        df['tf_pred'] = df['tf_pred'].astype(float)
        df['logi_pred'] = df['logi_pred'].astype(float)

        """
        todo:
        馬番ごとでかうんとしていく。
        """
        # gbm_pred, tf_pred, logi_pred
        df['new_mark_flag'] = '×'
        df['new_flag'] = 0

        # 0.5が1個以上のフラグ作成。△
        df['new_mark_flag'].mask((df['gbm_pred'] >= 0.5) | (df['tf_pred'] >= 0.5) | (df['logi_pred'] >= 0.5), '△',
                                 inplace=True)
        # # 0.5が2個以上のフラグ作成。〇
        df['new_mark_flag'].mask((df['gbm_pred'] >= 0.5) & (df['tf_pred'] >= 0.5), '〇', inplace=True)
        df['new_mark_flag'].mask((df['gbm_pred'] >= 0.5) & (df['logi_pred'] >= 0.5), '〇', inplace=True)
        df['new_mark_flag'].mask((df['logi_pred'] >= 0.5) & (df['tf_pred'] >= 0.5), '〇', inplace=True)
        # 0.5が3個以上のフラグ作成。◎
        df['new_mark_flag'].mask((df['gbm_pred'] >= 0.5) & (df['tf_pred'] >= 0.5) & (df['logi_pred'] >= 0.5), '◎',
                                 inplace=True)

        df['new_flag'].mask(((df['gbm_pred'] * 0.3) + (df['tf_pred'] * 0.6) + (df['logi_pred'] * 0.1)) >= 0.5, 1,
                            inplace=True)

        df = pd.merge(df_pred, df, on='raceid', how='left')

        return df
