import pandas as pd
import numpy as np


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
        df = df.drop('basetime', axis=1)

        df = df.replace({"class": {"1勝": "500万", "2勝": "1000万", "3勝": "1600万"}})
        df = df.fillna({'fathertype': 'その他のエクリプス系'})

        # 脚質の調整
        df = df.replace({'legtype': {'ﾏｸﾘ': '追込'}})
        df = df.replace({'legtype': {'後方': '追込'}})
        df = df.replace({'legtype': {'中団': '差し'}})
        df = df.fillna({'legtype': '自在'})

        # 競争除外馬を削除する。
        df = df.dropna(subset=['pop', 'odds', '3ftime'])

        df_main = df.copy()

        self.df = df
        self.df_main = df_main

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

    def jockey_data_process(self):
        df_start = self.df
        df = df_start.copy()

        df.loc[df['result'] >= 3, 'result'] = 0
        df.loc[df['result'] == 2, 'result'] = 1

        table_jockey = pd.pivot_table(df, index='jocky', columns='place', values='result', aggfunc='mean', dropna=False)
        table_jockey = table_jockey.fillna(0)

        table_jockey = pd.DataFrame(table_jockey)
        table_jockey = table_jockey.round(4)
        table_jockey = table_jockey.add_prefix('jockey_')

        return table_jockey

    def father_data_process(self, index='father'):
        df_start = self.df
        df = df_start.copy()
        # fatherの連結対象(レース場成績、距離、芝ダート、重馬場成績)

        df.loc[df['result'] >= 3, 'result'] = 0
        df.loc[df['result'] == 2, 'result'] = 1

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

        df['legtype'] = df['legtype'].map({'逃げ': 0, '先行': 1, '差し': 2, '追込': 3, '自在': 4})
        legtypes = df.groupby(index).legtype.apply(lambda x: x.mode()).reset_index()

        legtype = pd.DataFrame(legtypes)
        legtype['legtype'] = legtype['legtype'].map({0: '逃げ', 1: '先行', 2: '差し', 3: '追込', 4: '自在'})

        legtype = legtype.drop('level_1', axis=1)

        time_3f = df.groupby(index).mean()['3ftime']
        time3f = pd.DataFrame(time_3f)

        father = pd.merge(table_father1, legtype, on=index, how='left')
        father = pd.merge(father, time3f, on=index, how='left')

        father = father.round(3)
        father = father.add_prefix('{}_'.format(index))
        return father

    @staticmethod
    def pre_race_data_process(dataframe):
        df = dataframe

        name_days_df = df[["horsename", "days", "pop", "odds",
                           "rank3", "rank4", "3ftime", "result", 'speedindex']].sort_values(['horsename', 'days'])

        name_list = name_days_df['horsename'].unique()

        df_shift_list = []
        df_rolling_list = []

        df = df.drop('speedindex', axis=1)
        # 確率が低くなるのはresultmeanが原因となっている可能性大
        # →resultで平均をとるとおかしくなったのでtarget以外で特徴量を生成した方がいいかも

        # renamesurukoto
        for name in name_list:
            name_df = name_days_df[name_days_df['horsename'] == name]
            shift_name_df = name_df[["pop", "odds", "rank3", "rank4", "3ftime", "result", 'speedindex']].shift(1)
            rolling_name_df = name_df[["pop", "odds", "rank3", "rank4", "3ftime", 'speedindex']].rolling(5, min_periods=2)\
                .agg(['mean', 'max', 'min'])
            shift_name_df['horsename'] = name
            rolling_name_df['horsename'] = name

            df_shift_list.append(shift_name_df)
            df_rolling_list.append(rolling_name_df)

        df_sh_before = pd.concat(df_shift_list)
        df_ro_before = pd.concat(df_rolling_list)

        df_sh_before['days'] = name_days_df['days']
        df_ro_before['days'] = name_days_df['days']

        df_sh_before = df_sh_before.rename(columns={'pop': 'pre_pop', 'odds': 'pre_odds', 'rank3': 'pre_rank3',
                                                    'rank4': 'pre_rank4', '3ftime': 'pre_3ftime',
                                                    'result': 'pre_result'})

        df = pd.merge(df, df_sh_before, on=['horsename', 'days'], how='inner')
        df = pd.merge(df, df_ro_before, on=['horsename', 'days'], how='inner')

        df = df.dropna(subset=["speedindex"])

        return df

    def formatting_data_process(self):
        df_speed = self.speedindex_data_process()
        df_jockey = self.jockey_data_process()
        df_father = self.father_data_process()
        df_ftype = self.father_data_process(index='fathertype')

        df_father = df_father.rename(columns={'father_father': 'father'})
        df_ftype = df_ftype.rename(columns={'fathertype_fathertype': 'fathertype'})

        main_df = self.df_main
        main_df = pd.merge(main_df, df_speed, on=['raceid', 'days', 'horsename'], how='left')
        main_df = pd.merge(main_df, df_jockey, on='jocky', how='left')
        main_df = pd.merge(main_df, df_father, on='father', how='left')
        main_df = pd.merge(main_df, df_ftype, on='fathertype', how='left')

        main_df = main_df.dropna(how="any")

        main_df = self.pre_race_data_process(main_df)

        d_ranking = lambda x: 1 if x in [1, 2] else 0
        main_df['flag'] = main_df['result'].map(d_ranking)

        drop_list = ['rank3', 'rank4', '3ftime', 'time']
        main_df = main_df.drop(drop_list, axis=1)

        return main_df.to_csv('Keiba/datafile/pred_data/testcsvdataframe.csv', encoding='utf_8_sig', index=False)

    def add_feature_formatting_process(self):
        # self.formatting_data_process()

        df = pd.read_csv('Keiba/datafile/pred_data/testcsvdataframe.csv')

        df = df.dropna(how="any")
        df = df.rename(columns={"('speedindex', 'mean')": "speedmean", "('speedindex', 'max')": "speedmax", "('speedindex', 'min')": "speedmin",
                                "('pop', 'mean')": "popmean", "('pop', 'max')": "popmax", "('pop', 'min')": "popmin",
                                "('odds', 'mean')": "oddsmean", "('odds', 'max')": "oddsmax", "('odds', 'min')": "oddsmin",
                                "('rank3', 'mean')": "rank3mean", "('rank3', 'max')": "rank3max", "('rank3', 'min')": "rank3min",
                                "('rank4', 'mean')": "rank4mean", "('rank4', 'max')": "rank4max", "('rank4', 'min')": "rank4min",
                                "('3ftime', 'mean')": "3ftimemean", "('3ftime', 'max')": "3ftimemax", "('3ftime', 'min')": "3ftimemin",
                                })

        drop_list = ["rank3max", "rank3min", "rank4max", "rank4min"]
        df = df.drop(drop_list, axis=1)

        # 　特徴量生成
        df['odds_hi'] = (df['odds'] / df['pop'])
        df['re_odds_hi'] = (df['pre_odds'] / df['pre_pop'])
        df['odds_hi*2'] = df['odds_hi'] ** 2
        df['re_odds_hi*2'] = df['re_odds_hi'] ** 2
        df['re_3_to_4time'] = (df['pre_rank4'] - df['pre_rank3'])
        df['re_3_to_4time_hi*2'] = (df['pre_rank4'] / df['pre_rank3']) ** 2
        df['re_pop_now_pop'] = (df['pre_pop'] - df['pop'])
        df['re_odds_now_odds'] = (df['pre_odds'] - df['odds'])
        df['re_result_to_pop'] = (df['pre_result'] - df['pre_pop'])
        df['speedmax_speedmin'] = df['speedmax'] - df['speedmin']
        df['popmax_popmin'] = df['popmax'] - df['popmin']
        df['oddsmax_oddsmin'] = df['oddsmax'] - df['oddsmin']
        df['3ftimemax_3ftimemin'] = df['3ftimemax'] - df['3ftimemin']

        feature_list = ['odds_hi', 're_odds_hi', 'odds_hi*2', 're_odds_hi*2', 're_3_to_4time', 're_3_to_4time_hi*2',
                        're_pop_now_pop', 're_odds_now_odds', 're_result_to_pop', 'speedmax_speedmin', 'popmax_popmin',
                        'popmax_popmin', 'oddsmax_oddsmin', '3ftimemax_3ftimemin']

        for feature in feature_list:
            df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
            df[feature] = df[feature].fillna(0)

        return df

