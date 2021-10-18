from datetime import datetime

import pandas as pd
import optuna.integration.lightgbm as lgb

from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split


class KeibaPrediction:

    def __init__(self, data):
        self.data = data

    def gbm_params_keiba(self):
        df = self.data

        df['days'] = pd.to_datetime(df['days'])

        df_pred = df[df['days'] > datetime(2021, 10, 16)]
        df_pred_droped = df_pred.drop(['flag', 'days', 'horsename', 'raceid', 'odds', 'pop'], axis=1)

        df = df[df['days'] <= datetime(2021, 10, 16)]
        df = df.dropna(how='any')
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
            num_boost_round=100,
            early_stopping_rounds=20,
        )

        predict_proba = model.predict(df_pred_droped, num_iteration=model.best_iteration)

        predict = [0 if i < 0.5 else 1 for i in predict_proba]

        predict = pd.DataFrame({"raceid": df_pred['raceid'],
                                "pred": predict})

        predict_df = pd.merge(df_pred, predict, on='raceid', how='left')
        predict_df = predict_df.dropna(how='any')

        return predict_df