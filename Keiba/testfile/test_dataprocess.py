import dask.dataframe as dd
import gc
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

"""
Memo:
numpy配列を加工すると処理が急激に重くなるので(16GB一杯になる)、解決できるまでは主成分分析・クラスター分析
などといったreturnがnumpy配列を返す分析手法であるものは使用不可とする。解決が出来次第処理に追加する。
※現状出ている解決策
・メモリを増やす。
・AWSなどを利用する。
"""


class TestDataProcess:

    def __init__(self, csv_data):
        df = pd.read_csv(csv_data, encoding='shift-jis')

        # 簡単な削除
        # 障害レースを削除
        race_dis = [1000, 1200, 1400, 1500, 1600, 1700, 1800, 2000, 2200,
                    2300, 2400, 2500, 2600, 3000, 3200, 3400, 3600]
        df = df[df['distance'].isin(race_dis)]

        # １部欠損値補完・除外
        df = df[~df['result'].isin([0])]

        df = df.replace({"class": {"1勝": "500万", "2勝": "1000万", "3勝": "1600万"}})
        df = df.fillna({'fathertype': 'その他のエクリプス系'})

        # 脚質の調整
        df = df.replace({'legtype': {'ﾏｸﾘ': '追込'}})
        df = df.replace({'legtype': {'後方': '追込'}})
        df = df.replace({'legtype': {'中団': '差し'}})
        df = df.fillna({'legtype': '自在'})

        # 競争除外馬を削除する。
        df = df.dropna(subset=['pop', 'odds', '3ftime'])

        df['days'] = pd.to_datetime(df['days'])
        df_main = df.copy()

        self.df = df
        self.df_main = df_main

    @staticmethod
    def reduce_mem_usage(df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                    start_mem - end_mem) / start_mem))
        return df

    def speedindex_data_process(self):
        df_start = self.df
        df = df_start.copy()

        df_basetime = df.copy()
        df_basetime = df_basetime[(df_basetime['result'] <= 3)]

        base_time = df_basetime.groupby(['place', 'distance', 'turf']).mean()['time'].reset_index()
        base_time = base_time.rename(columns={'time': 'basetime'})
        base_time['basetime'] = base_time['basetime'].round(1)

        df = pd.merge(df, base_time, on=['place', 'turf', 'distance'], how='left')

        df['conditionindex'] = 0

        df['conditionindex'].mask((df['condition'] == '良') & (df['turf'] == '芝'), -25, inplace=True)
        df['conditionindex'].mask((df['condition'] == '稍') & (df['turf'] == '芝'), -15, inplace=True)
        df['conditionindex'].mask((df['condition'] == '重') & (df['turf'] == '芝'), -5, inplace=True)
        df['conditionindex'].mask((df['condition'] == '良') & (df['turf'] == 'ダ'), -20, inplace=True)
        df['conditionindex'].mask((df['condition'] == '稍') & (df['turf'] == 'ダ'), -10, inplace=True)

        df['basetime'].fillna(df['basetime'].median())
        df['weight'].fillna(df['weight'].median())

        time = (df['basetime'] * 10) - (df['time'] * 10)
        disindex = 1 / (df['basetime'] * 10) * 1000
        weight = (df['weight'] - 55) * 2

        df['speedindex'] = time * disindex + df['conditionindex'] + weight + 80

        df_speedindex = df[['raceid', 'days', 'horsename', 'speedindex']]

        return df_speedindex

    def last_race_index(self):

        df = self.df_main.copy()
        # 「(頭数-着順) + (人気-着順)*クラス指数」
        # 着順＝着順*(18/頭数), 人気=人気*(18/頭数)
        tyakuzyun = df['result'] * (18 / df['horsecount'])
        ninki = df['pop'] / (18 / df['horsecount'])
        df['last_race_index'] = ((18 - tyakuzyun) + (18 - ninki))
        df['class_index'] = 1.2

        df['class_index'].mask((df['class'] == '未勝利') | (df['class'] == '新馬'), 0.8, inplace=True)
        df['class_index'].mask((df['class'] == '500万'), 0.9, inplace=True)
        df['class_index'].mask((df['class'] == '1000万'), 1.0, inplace=True)
        df['class_index'].mask((df['class'] == '1600万'), 1.1, inplace=True)

        df['last_race_index'] *= df['class_index']

        df_last_race_index = df[['raceid', 'days', 'horsename', 'last_race_index']]

        return df_last_race_index

    def jockey_data_process(self):
        df_start = self.df
        df = df_start.copy()
        df = df[df['days'] < datetime(2021, 1, 1)]

        df.loc[df['result'] >= 4, 'result'] = 0
        df.loc[(df['result'] <= 3) & (df['result'] >= 1), 'result'] = 1

        # 各ジョッキーの連対率（2021年1月1日まで集計対象）
        table_jockey = pd.pivot_table(df, index='jocky', columns='place', values='result', aggfunc='mean', dropna=False)
        table_jockey = table_jockey.fillna(0)

        means_jockey = df.groupby('jocky').mean()['result']

        table_jockey = pd.DataFrame(table_jockey)
        table_jockey = table_jockey.add_prefix('jockey_')

        table_jockey = pd.merge(table_jockey, means_jockey, how='left', on='jocky')
        table_jockey = table_jockey.rename(columns={"result": "resultmean"})
        table_jockey = table_jockey.round(4)

        # 主成分分析：次元削除
        pca = PCA()
        pca.fit(table_jockey)
        df_score = pd.DataFrame(pca.transform(table_jockey), index=table_jockey.index)

        df_score = df_score.loc[:, :5]
        df_score = pd.DataFrame(data=df_score)
        df_score = df_score.rename(columns={0: 'jockey_pca1', 1: 'jockey_pca2', 2: 'jockey_pca3',
                                            3: 'jockey_pca4', 4: 'jockey_pca5', 5: 'jockey_pca6',
                                            6: 'jockey_pca7', 7: 'jockey_pca8'})

        # TargetEncoding：HoldOut
        df_jockey = df[['jocky', 'result']]
        agg_df = df_jockey.groupby('jocky').agg({'result': ['sum', 'count']})

        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        ts = pd.Series(np.empty(df_jockey.shape[0]), index=df_jockey.index)

        for _, holdout_idx in folds.split(df_jockey, df_jockey.result):
            holdout_df = df_jockey.iloc[holdout_idx]
            holdout_agg_df = holdout_df.groupby('jocky').agg({'result': ['sum', 'count']})
            train_agg_df = agg_df - holdout_agg_df
            oof_ts = holdout_df.apply(
                lambda row: train_agg_df.loc[row.jocky][('result', 'sum')] / (train_agg_df.loc[row.jocky][
                    ('result', 'count')]+ 1), axis=1)
            ts[oof_ts.index] = oof_ts

        ts.name = 'holdout_ts_jockey'
        df_jockey = df_jockey.join(ts)
        df_jockey = df_jockey.drop('result', axis=1)

        df_jockey = pd.merge(df_jockey, df_score, how='left', on='jocky')


        cols = ['jockey_pca1', 'jockey_pca2', 'jockey_pca3',
                  'jockey_pca4', 'jockey_pca5', 'jockey_pca6', 'holdout_ts_jockey']


        del df_start, df, df_score, table_jockey, ts, holdout_df, holdout_agg_df, agg_df, train_agg_df, oof_ts
        gc.collect()

        df_new = df_jockey.groupby('jocky').agg('mean')
        df_new = df_new.round(3)

        return df_new

    def father_data_process(self, index='father'):
        df_start = self.df
        df = df_start.copy()
        df = df[df['days'] < datetime(2021, 1, 1)]

        df.loc[df['result'] >= 4, 'result'] = 0
        df.loc[(df['result'] <= 3) & (df['result'] >= 1), 'result'] = 1

        # fatherの連結対象(レース場成績、距離、芝ダート、重馬場成績)（2021年5月31日まで集計対象）
        table_father_place = pd.pivot_table(df, index=index, columns='place', values='result', aggfunc='mean',
                                            dropna=False)
        table_father_distance = pd.pivot_table(df, index=index, columns='distance', values='result', aggfunc='mean',
                                               dropna=False)
        table_father_turf = pd.pivot_table(df, index=index, columns='turf', values='result', aggfunc='mean',
                                           dropna=False)
        table_father_condition = pd.pivot_table(df, index=index, columns='condition', values='result', aggfunc='mean',
                                                dropna=False)

        table_father = pd.merge(table_father_place, table_father_distance, on=index, how='left')
        table_father = pd.merge(table_father, table_father_turf, on=index, how='left')
        table_father = pd.merge(table_father, table_father_condition, on=index, how='left')

        table_father1 = table_father.fillna(0)

        time_3f = df.groupby(index).mean()['3ftime']
        result_mean = df.groupby(index).mean()['result']
        time3f = pd.DataFrame(time_3f)
        result_mean = pd.DataFrame(result_mean)
        result_mean = result_mean.rename(columns={"result": "resultmean"})

        father = pd.merge(table_father1, time3f, on=index, how='left')
        father = pd.merge(father, result_mean, on=index, how='left')

        father = father.round(3)
        father = father.add_prefix('{}_'.format(index))

        # 主成分分析：次元削除
        pca = PCA()
        pca.fit(father)
        df_score = pd.DataFrame(pca.transform(father), index=father.index)

        df_score = df_score.loc[:, :8]
        df_score = pd.DataFrame(data=df_score)

        df_score = df_score.rename(columns={0: 'father_pca1', 1: 'father_pca2', 2: 'father_pca3', 3:'father_pca4',
                                            4: 'father_pca5', 5: 'father_pca6', 6: 'father_pca7', 7: 'father_pca8',
                                            8: 'father_pca9'})

        # TargetEncoding：HoldOut
        df_father = df[['father', 'result']]
        agg_df = df_father.groupby('father').agg({'result': ['sum', 'count']})

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        ts = pd.Series(np.empty(df_father.shape[0]), index=df_father.index)

        for _, holdout_idx in folds.split(df_father, df_father.result):
            holdout_df = df_father.iloc[holdout_idx]
            holdout_agg_df = holdout_df.groupby('father').agg({'result': ['sum', 'count']})
            train_agg_df = agg_df - holdout_agg_df
            oof_ts = holdout_df.apply(
                lambda row: train_agg_df.loc[row.father][('result', 'sum')] / (train_agg_df.loc[row.father][
                    ('result', 'count')]+ 1), axis=1)
            ts[oof_ts.index] = oof_ts

        ts.name = 'holdout_ts_father'
        df_father = df_father.join(ts)
        df_father = df_father.drop('result', axis=1)

        df_father = pd.merge(df_father, df_score, how='left', on='father')

        cols = ['father_pca1', 'father_pca2', 'father_pca3', 'father_pca4', 'father_pca5', 'father_pca6',
                'father_pca7', 'father_pca8', 'father_pca9', 'holdout_ts_father']

        del df_start, df,df_score, table_father, ts, holdout_df, holdout_agg_df, agg_df, train_agg_df, oof_ts
        gc.collect()

        df_new = df_father.groupby('father').agg('mean')
        df_new = df_new.round(3)

        return df_new

    def fathermon_data_process(self, index='fathermon'):
        df_start = self.df
        df = df_start.copy()
        df = df[df['days'] < datetime(2021, 1, 1)]

        df.loc[df['result'] >= 4, 'result'] = 0
        df.loc[(df['result'] <= 3) & (df['result'] >= 1), 'result'] = 1

        # fatherの連結対象(レース場成績、距離、芝ダート、重馬場成績)（2021年5月31日まで集計対象）
        table_father_place = pd.pivot_table(df, index=index, columns='place', values='result', aggfunc='mean',
                                            dropna=False)
        table_father_distance = pd.pivot_table(df, index=index, columns='distance', values='result', aggfunc='mean',
                                               dropna=False)
        table_father_turf = pd.pivot_table(df, index=index, columns='turf', values='result', aggfunc='mean',
                                           dropna=False)
        table_father_condition = pd.pivot_table(df, index=index, columns='condition', values='result', aggfunc='mean',
                                                dropna=False)

        table_father = pd.merge(table_father_place, table_father_distance, on=index, how='left')
        table_father = pd.merge(table_father, table_father_turf, on=index, how='left')
        table_father = pd.merge(table_father, table_father_condition, on=index, how='left')

        table_father1 = table_father.fillna(0)

        time_3f = df.groupby(index).mean()['3ftime']
        result_mean = df.groupby(index).mean()['result']
        time3f = pd.DataFrame(time_3f)
        result_mean = pd.DataFrame(result_mean)
        result_mean = result_mean.rename(columns={"result": "resultmean"})

        father = pd.merge(table_father1, time3f, on=index, how='left')
        father = pd.merge(father, result_mean, on=index, how='left')

        father = father.round(3)
        father = father.add_prefix('{}_'.format(index))

        # 主成分分析：次元削除
        pca = PCA()
        pca.fit(father)
        df_score = pd.DataFrame(pca.transform(father), index=father.index)

        df_score = df_score.loc[:, :12]
        df_score = pd.DataFrame(data=df_score)

        df_score = df_score.rename(columns={0: 'fathermon_pca1', 1: 'fathermon_pca2', 2: 'fathermon_pca3', 3: 'fathermon_pca4',
                                            4: 'fathermon_pca5', 5: 'fathermon_pca6', 6: 'fathermon_pca7', 7: 'fathermon_pca8',
                                            8: 'fathermon_pca9', 9: 'fathermon_pca10', 10: 'fathermon_pca11', 11: 'fathermon_pca12',
                                            12: 'fathermon_pca13'})

        # TargetEncoding：HoldOut
        df_father = df[['fathermon', 'result']]
        agg_df = df_father.groupby('fathermon').agg({'result': ['sum', 'count']})

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        ts = pd.Series(np.empty(df_father.shape[0]), index=df_father.index)

        for _, holdout_idx in folds.split(df_father, df_father.result):
            holdout_df = df_father.iloc[holdout_idx]
            holdout_agg_df = holdout_df.groupby('fathermon').agg({'result': ['sum', 'count']})
            train_agg_df = agg_df - holdout_agg_df
            oof_ts = holdout_df.apply(
                lambda row: train_agg_df.loc[row.fathermon][('result', 'sum')] / (train_agg_df.loc[row.fathermon][
                    ('result', 'count')]+ 1), axis=1)
            ts[oof_ts.index] = oof_ts

        ts.name = 'holdout_ts_fathermon'
        df_father = df_father.join(ts)
        df_father = df_father.drop('result', axis=1)

        df_father = pd.merge(df_father, df_score, how='left', on='fathermon')

        cols = ['fathermon_pca1', 'fathermon_pca2', 'fathermon_pca3', 'fathermon_pca4', 'fathermon_pca5', 'fathermon_pca6',
                'fathermon_pca7', 'fathermon_pca8', 'fathermon_pca9', 'fathermon_pca10', 'fathermon_pca11', 'fathermon_pca12',
                'fathermon_pca13', 'holdout_ts_fathermon']

        df_father[cols] = df_father[cols].astype('float32')

        del df_start, df,df_score, table_father, ts, holdout_df, holdout_agg_df, agg_df, train_agg_df, oof_ts
        gc.collect()

        df_new = df_father.groupby('fathermon').agg('mean')
        df_new = df_new.round(3)

        return df_new

    @staticmethod
    def pre_race_data_process(dataframe):
        df = dataframe
        df['count'] = 1
        df['rentai'] = (df['result'] <= 3).astype(int)

        name_days_df = df[["horsename", "place", "turf", "distance", "days", "pop", "odds",
                           "rank3", "rank4", "3ftime", "result", 'speedindex', 'last_race_index', 'count', 'rentai']].sort_values(['horsename', 'days'])

        name_list = name_days_df['horsename'].unique()

        df_shift_list = []
        df_rolling_list = []

        df = df.drop(['speedindex', 'last_race_index'], axis=1)
        # 確率が低くなるのはresultmeanが原因となっている可能性大
        # →resultで平均をとるとおかしくなったのでtarget以外で特徴量を生成した方がいいかも
        agg_list = {
            "pop": ['mean', 'max', 'min'],
            "odds": ['mean', 'max', 'min'],
            "3ftime": ['mean', 'max', 'min'],
            "speedindex": ['mean', 'max', 'min'],
            "result": ['mean'],
            "count": ['sum'],
            "rentai": ['sum'],
        }

        # renamesurukoto
        for name in name_list:
            name_df = name_days_df[name_days_df['horsename'] == name]
            shift_name_df = name_df[["place", "turf", "distance", "pop", "odds", "rank3",
                                     "rank4", "3ftime", "result", 'speedindex', 'last_race_index']].shift(1)
            rolling_name_df = name_df[["pop", "odds", "3ftime", 'speedindex', "result", 'count', 'rentai']].rolling(5, min_periods=1)\
                .agg(agg_list).shift(1)
            shift_name_df['horsename'] = name
            rolling_name_df['horsename'] = name

            df_shift_list.append(shift_name_df)
            df_rolling_list.append(rolling_name_df)

        df_sh_before = pd.concat(df_shift_list)
        df_ro_before = pd.concat(df_rolling_list)

        df_sh_before['days'] = name_days_df['days']
        df_ro_before['days'] = name_days_df['days']

        df_sh_before = df_sh_before.rename(columns={'place': 'pre_place', 'turf': 'pre_turf', 'distance': 'pre_distance',
                                                    'pop': 'pre_pop', 'odds': 'pre_odds', 'rank3': 'pre_rank3',
                                                    'rank4': 'pre_rank4', '3ftime': 'pre_3ftime',
                                                    'result': 'pre_result'})

        df = pd.merge(df, df_sh_before, on=['horsename', 'days'], how='inner')
        df = pd.merge(df, df_ro_before, on=['horsename', 'days'], how='inner')

        df = df.dropna(subset=["speedindex"])

        return df

    def formatting_data_process(self):
        print("基礎処理スタート")
        df_speed = self.speedindex_data_process()
        df_lastrace = self.last_race_index()

        main_df = self.df_main
        main_df = pd.merge(main_df, df_speed, on=['raceid', 'days', 'horsename'], how='left')
        main_df = pd.merge(main_df, df_lastrace, on=['raceid', 'days', 'horsename'], how='left')

        del df_speed
        del df_lastrace
        gc.collect()
        print("終了")
        time.sleep(10)

        print('長距離開始')
        main_df = self.pre_race_data_process(main_df)
        main_df = self.reduce_mem_usage(main_df)
        print("終了")
        time.sleep(10)

        print("データ入力スタート")
        d_ranking = lambda x: 1 if x in [1, 2, 3] else 0
        main_df['flag'] = main_df['result'].map(d_ranking)

        drop_list = ['rank3', 'rank4', '3ftime', 'time']
        main_df = main_df.drop(drop_list, axis=1)

        del d_ranking
        gc.collect()

        print("終了")
        print("データマージスタート")
        print("騎手データマージ")

        df_jockey = self.jockey_data_process()

        main_df = dd.merge(main_df, df_jockey, on='jocky', how='left')

        del df_jockey
        gc.collect()

        print("終了")
        time.sleep(10)

        print("父馬データマージ")

        df_father = self.father_data_process()

        df_father = df_father.rename(columns={'father_father': 'father'})
        main_df = dd.merge(main_df, df_father, on='father', how='left')

        del df_father
        gc.collect()
        print("終了")

        print("母父馬データマージ")
        df_father_mon = self.fathermon_data_process()

        df_father_mon = df_father_mon.rename(columns={'father_fathermon': 'fathermon'})
        main_df = dd.merge(main_df, df_father_mon, on='fathermon', how='left')

        del df_father_mon
        gc.collect()

        print("完了！！！！")

        return main_df.to_csv('Keiba/datafile/pred_data/testcsvdataframe.csv', encoding='utf_8_sig', index=False)

    def add_feature_formatting_process(self, switch=True):
        if switch:
            self.formatting_data_process()

        df = pd.read_csv('Keiba/datafile/pred_data/testcsvdataframe.csv')

        fillna_list = df.loc[:, 'holdout_ts_jockey':].columns.tolist()

        for fillna_col in fillna_list:
            df[fillna_col] = df[fillna_col].fillna(df[fillna_col].mean())

        df = df.rename(columns={"('speedindex', 'max')": "speedmax", "('speedindex', 'mean')": "speedmean", "('speedindex', 'min')": "speedmin",
                                "('pop', 'mean')": "popmean", "('pop', 'max')": "popmax", "('pop', 'min')": "popmin",
                                "('odds', 'mean')": "oddsmean", "('odds', 'max')": "oddsmax", "('odds', 'min')": "oddsmin",
                                "('3ftime', 'min')": "3ftimemin", "('3ftime', 'mean')": "3ftimemean", "('3ftime', 'max')": "3ftimemax",
                                "('count', 'sum')": "count5sum", "('rentai', 'sum')": "rentai5sum", "('result', 'mean')": 'resultmean'
                                })

        # 　特徴量生成
        df['flag_konkan'] = (df['distance'] % 400 == 0).astype(int)
        df['flag_pre_konkan'] = (df['pre_distance'] % 400 == 0).astype(int)
        df['odds_hi'] = (df['odds'] / df['pop'])
        df['re_odds_hi'] = (df['pre_odds'] / df['pre_pop'])
        df['odds_hi*2'] = df['odds_hi'] ** 2
        df['re_odds_hi*2'] = df['re_odds_hi'] ** 2
        df['re_3_to_4time'] = (df['pre_rank4'] - df['pre_rank3'])
        df['re_3_to_4time_hi*2'] = (df['pre_rank4'] / df['pre_rank3']) ** 2
        df['re_pop_now_pop'] = (df['pre_pop'] - df['pop'])
        df['re_odds_now_odds'] = (df['pre_odds'] - df['odds'])
        df['re_result_to_pop'] = (df['pre_result'] - df['pre_pop'])
        df['popmax_popmin'] = df['popmax'] - df['popmin']
        df['oddsmax_oddsmin'] = df['oddsmax'] - df['oddsmin']
        df['rentai_ritu'] = ((df["rentai5sum"] / df["count5sum"])).round(3)

        feature_list = ['odds_hi', 're_odds_hi', 'odds_hi*2', 're_odds_hi*2', 're_3_to_4time', 're_3_to_4time_hi*2',
                        're_pop_now_pop', 're_odds_now_odds', 're_result_to_pop', 'popmax_popmin', 'popmax_popmin', 'rentai_ritu']

        for feature in feature_list:
            df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
            df[feature] = df[feature].fillna(0)

        drop_list_cols = ['odds', 'pop', "rentai5sum", "count5sum", 'count', 'rentai']
        df = df.drop(drop_list_cols, axis=1)

        re_fillna_list = df.loc[:, 'popmean': 'speedmean'].columns.tolist()
        for fillna_col in re_fillna_list:
            df[fillna_col] = df[fillna_col].fillna(df[fillna_col].mean())

        df = df.dropna(how="any")

        return df

