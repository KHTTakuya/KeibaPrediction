import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow import feature_column
from sklearn.preprocessing import StandardScaler

from Keiba import datalist


class KeibaProcessing:

    def __init__(self, csv_data, pred_data=None):
        """
        :param csv_data:
        このクラスに競馬の前処理方法を記載すること、継ぎ足しする場合は
        "create_dataframe"に実行関数を記載する。
        """
        self.csv_data = csv_data
        self.pred_data = pred_data

    def create_dataframe(self):
        """
        :return: df(pandas:dataframe)
        create_dataframeからlightGBMに行く場合は、data_feature_and_formating関数を実行してから起動すること。
        また、Tensorflowに行く場合は、df_to_tflayerとdata_feature_and_formatingをFalseにしてから実行すること。
        なお原則、データ加工を行う際はデバック作業、新規実装を除いてこちらの関数のみを利用すること。
        """
        df_data = self.preprocessing(self.csv_data)

        main = self.pre_horse_data_process(df_data, pred_data=self.pred_data)
        jockey = self.jockey_data_process(df_data)
        father = self.father_data_process(df_data)
        father_type = self.father_data_process(df_data, index='fathertype')
        place = self.place_data_process(df_data)
        distance = self.place_data_process(df_data, index='distance')

        df = self.data_concatenation(main, jockey, father, father_type, place, distance)

        return df

    @staticmethod
    def preprocessing(data):
        """
        :param data: csvデータ
        :return: df(pandas:dataframe)
        基礎条件のデータ削除
        """
        df = pd.read_csv(data, encoding="shift-jis")
        # 障害レースを削除する。
        race_dis = [1000, 1200, 1400, 1500, 1600, 1700, 1800, 2000, 2200,
                    2300, 2400, 2500, 2600, 3000, 3200, 3400, 3600]

        df01 = df[df['distance'].isin(race_dis)]

        # スピード指数の作成
        df01['conditionindex'] = 0

        df01['conditionindex'].mask((df01['condition'] == '良') & (df01['turf'] == '芝'), -25, inplace=True)
        df01['conditionindex'].mask((df01['condition'] == '稍') & (df01['turf'] == '芝'), -15, inplace=True)
        df01['conditionindex'].mask((df01['condition'] == '重') & (df01['turf'] == '芝'), -5, inplace=True)
        df01['conditionindex'].mask((df01['condition'] == '良') & (df01['turf'] == 'ダ'), -20, inplace=True)
        df01['conditionindex'].mask((df01['condition'] == '稍') & (df01['turf'] == 'ダ'), -10, inplace=True)

        df['basetime'].fillna(df['basetime'].median())
        df['weight'].fillna(df['weight'].median())

        time = (df01['basetime'] * 10) - (df01['time'] * 10)
        disindex = 1 / (df01['basetime'] * 10) * 1000
        weight = (df01['weight'] - 55) * 2

        df01['speedindex'] = time * disindex + df01['conditionindex'] + weight + 80

        df01 = df01.drop(['basetime', 'weight', 'conditionindex'], axis=1)

        # 脚質の調整

        df01 = df01.replace({'legtype': {'ﾏｸﾘ': '追込'}})
        df01 = df01.replace({'legtype': {'後方': '追込'}})
        df01 = df01.replace({'legtype': {'中団': '差し'}})
        df01 = df01.fillna({'legtype': '自在'})

        df01 = df01.replace({'distance': [1000, 1200, 1400, 1500]}, '短距離')
        df01 = df01.replace({'distance': [1600, 1700, 1800]}, 'マイル')
        df01 = df01.replace({'distance': [2000, 2200, 2300, 2400]}, '中距離')
        df01 = df01.replace({'distance': [2500, 2600, 3000, 3200, 3400, 3600]}, '長距離')

        return df01

    @staticmethod
    def jockey_data_process(data):
        """
        :param data: preprocessingから受け取る。
        :return: df(pandas:dataframe)
        騎手のdataframeを作成する。
        """
        df = data

        df.loc[df['result'] >= 3, 'result'] = 0
        df.loc[df['result'] == 2, 'result'] = 1

        table_jockey = pd.pivot_table(df, index='jocky', columns='place', values='result', aggfunc='mean', dropna=False)
        table_jockey = table_jockey.fillna(0)

        table_jockey = pd.DataFrame(table_jockey)
        table_jockey = table_jockey.round(4)
        table_jockey = table_jockey.add_prefix('jockey_')

        return table_jockey

    @staticmethod
    def father_data_process(data, index='father'):
        """
        :param data: df(preprocessingを通して)いれること。csvデータのまま入れるとエラーがおきる。
        :param index: 初期はfather,fathertypeで作成する場合はfathertypeに変更すること。
        :return: df(dataframe)
        indexにはfatherとfathertypeのみが対象。その他を入れるのは予期せぬエラーが起きる可能があるため入れないこと。
        placeとdistanceで作成する場合は、place_data_processで作成すること。
        """
        df = data
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
    def place_data_process(data, index='place'):
        """
        :param data: df(preprocessingを通して)いれること。csvデータのまま入れるとエラーがおきる。
        :param index: 初期はplace,distanceで作成する場合はdistanceに変更すること。
        :return: df(pandas:dataframe)
        indexにはplaceとdistanceのみが対象。その他を入れるのは予期せぬエラーが起きる可能があるため入れないこと。
        fatherとfathertypeで作成したい場合は、father_data_processで作成すること。
        """
        global table_place_time
        df = data

        df.loc[df['result'] >= 3, 'result'] = 0
        df.loc[df['result'] == 2, 'result'] = 1

        df = df.query('result == 1')

        if index == 'distance':
            table_place_time = pd.pivot_table(df, index=index, values='time', aggfunc='mean')

        table_place_rank3 = pd.pivot_table(df, index=index, values='rank3', aggfunc='mean')
        table_place_rank4 = pd.pivot_table(df, index=index, values='rank4', aggfunc='mean')
        table_place_3ftime = pd.pivot_table(df, index=index, values='3ftime', aggfunc='mean')
        table_place_pop = pd.pivot_table(df, index=index, values='pop', aggfunc='mean')
        table_place_odds = pd.pivot_table(df, index=index, values='odds', aggfunc='mean')

        table_place = pd.merge(table_place_odds, table_place_pop, on=index, how='left')

        if index == 'distance':
            table_place = pd.merge(table_place, table_place_time, on=index, how='left')
            table_place = pd.merge(table_place, table_place_rank3, on=index, how='left')
            table_place = pd.merge(table_place, table_place_rank4, on=index, how='left')
            table_place = pd.merge(table_place, table_place_3ftime, on=index, how='left')
        else:
            table_place = pd.merge(table_place, table_place_rank3, on=index, how='left')
            table_place = pd.merge(table_place, table_place_rank4, on=index, how='left')
            table_place = pd.merge(table_place, table_place_3ftime, on=index, how='left')

        place = pd.DataFrame(table_place)
        place = place.round(3)

        place = place.add_prefix('{}_'.format(index))

        return place

    @staticmethod
    def pre_horse_data_process(data, pred_data=None):
        """
        :param data: df(preprocessingを通して)いれること。csvデータのまま入れるとエラーがおきる。
        :param pred_data: 原則Noneにすること(こちらは後日修正を行う。)
        :return: df(pandas:dataframe)
        前走の出走データを作成する。これが基本のデータになる。
        pd.mergeする際はこのデータを中心にmergeすること。
        """
        if pred_data is not None:
            df = pd.concat(data, pred_data)
        else:
            df = data

        df = df.dropna(how='any')

        df['days'] = pd.to_datetime(df['days'])
        name_days_df = df[["horsename", "days", "pop",
                           "odds", "rank3", "rank4", "3ftime", "result", 'speedindex']].sort_values(
            ['horsename', 'days'])

        name_list = name_days_df['horsename'].unique()

        df_list = []
        df = df.drop('speedindex', axis=1)

        for name in name_list:
            name_df = name_days_df[name_days_df['horsename'] == name]
            shift_name_df = name_df[["pop", "odds", "rank3", "rank4", "3ftime", "result", 'speedindex']].shift(1)
            shift_name_df['horsename'] = name
            df_list.append(shift_name_df)

        df_before = pd.concat(df_list)
        df_before['days'] = name_days_df['days']

        df_before = df_before.rename(columns={'pop': 'pre_pop', 'odds': 'pre_odds', 'rank3': 'pre_rank3',
                                              'rank4': 'pre_rank4', '3ftime': 'pre_3ftime', 'result': 'pre_result'})

        df = pd.merge(df, df_before, on=['horsename', 'days'], how='inner')

        return df

    @staticmethod
    def data_concatenation(main, jockey, father, father_type, place, distance):
        """
        :param main: df(pandas:dataframe) pre_horse_data_processで作成したデータ。
        :param jockey: df(pandas:dataframe) preprocessingで作成したデータ。
        :param father: df(pandas:dataframe) father_data_processで作成したデータ。
        :param father_type: df(pandas:dataframe) father_data_processで作成したデータ。(index='fathertype')
        :param place: df(pandas:dataframe) place_data_processで作成したデータ。
        :param distance: df(pandas:dataframe) place_data_processで作成したデータ。(index='distance')
        :return: df(pandas:dataframe)
        以下paramsを連結させる。
        """
        df = main

        father = father.rename(columns={'father_father': 'father'})
        father_type = father_type.rename(columns={'fathertype_fathertype': 'fathertype'})

        df = pd.merge(df, jockey, on='jocky', how='left')
        df = pd.merge(df, father, on='father', how='left')
        df = pd.merge(df, father_type, on='fathertype', how='left')
        df = pd.merge(df, place, on='place', how='left')
        df = pd.merge(df, distance, on='distance', how='left')

        df = df.drop('fathertype_legtype', axis=1)

        return df

    @staticmethod
    def data_feature_and_formating(processed_data, gbmflag=True):
        """
        :param processed_data: df(pandas:dataframe)data_concatenationから持ってくること。
        :param gbmflag: True(default)の場合はLightGBM用にデータに加工される。
        Falseの場合はtensorflow用データに加工される。
        :return: df(pandas:dataframe)
        """
        df = processed_data

        d_ranking = lambda x: 1 if x in [1, 2] else 0
        df['flag'] = df['result'].map(d_ranking)

        drop_list = ['result', 'rank3', 'rank4', '3ftime', 'time']
        df = df.drop(drop_list, axis=1)

        cat_cols = ['place', 'class', 'turf', 'distance', 'weather', 'condition', 'sex', 'father', 'mother',
                    'fathertype', 'fathermon', 'legtype', 'jocky', 'trainer', 'father_legtype']

        # 　特徴量生成
        df['odds_hi'] = (df['odds'] / df['pop'])
        df['re_odds_hi'] = (df['pre_odds'] / df['pre_pop'])
        df['odds_hi*2'] = df['odds_hi'] ** 2
        df['re_odds_hi*2'] = df['re_odds_hi'] ** 2
        df['re_3_to_4time'] = (df['pre_rank4'] - df['pre_rank3'])
        df['re_3_to_4time_hi*2'] = (df['pre_rank4'] / df['pre_rank3']) ** 2
        df['father_3f_to_my'] = (df['father_3ftime'] - df['pre_3ftime'])
        df['fathertype_3f_to_my'] = (df['fathertype_3ftime'] - df['pre_3ftime'])
        df['re_pop_now_pop'] = (df['pre_pop'] - df['pop'])
        df['re_odds_now_odds'] = (df['pre_odds'] - df['odds'])
        df['re_result_to_pop'] = (df['pre_result'] - df['pre_pop'])

        feature_list = ['odds_hi', 're_odds_hi', 'odds_hi*2', 're_odds_hi*2', 're_3_to_4time', 're_3_to_4time_hi*2',
                        'father_3f_to_my', 'fathertype_3f_to_my', 're_pop_now_pop', 're_odds_now_odds',
                        're_result_to_pop']

        for feature in feature_list:
            df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
            df[feature] = df[feature].fillna(0)

        if gbmflag:
            for c in cat_cols:
                le = LabelEncoder()
                le.fit(df[c])
                df[c] = le.transform(df[c])
        else:
            previous_list = datalist.re_rename_list
            after_list = datalist.rename_list
            for i in range(len(previous_list)):
                df = df.rename(columns={previous_list[i]: after_list[i]})

            num_data = datalist.num_datas
            num_data.remove('horsenum')
            num_data.remove('speedindex')

            scaler = StandardScaler()
            sc = scaler.fit(df[num_data])

            scalered_df = pd.DataFrame(sc.transform(df[num_data]), columns=num_data, index=df.index)
            df.update(scalered_df)

        return df

    @staticmethod
    def df_to_tfdata(df_data):
        """
        :param df_data: df(pandas:dataframe)data_feature_and_formatingから持ってくること。
        :return: featuer_layer(tensorflow用)
        tensorflow用に特徴量のデータ加工を行う。
        """
        df = df_data

        feature_columns = []

        num_data = datalist.num_datas

        for header in num_data:
            feature_columns.append(feature_column.numeric_column(header))

        horsenum = feature_column.numeric_column('horsenum')
        horsenum_buckets = feature_column.bucketized_column(horsenum, [2, 4, 6, 8, 10, 12, 14, 16, 18])
        feature_columns.append(horsenum_buckets)

        cat_data = ['place', 'class', 'turf', 'weather', 'condition', 'sex', 'father', 'mother', 'fathermon',
                    'fathertype', 'legtype', 'jocky', 'trainer']

        for cat in cat_data:
            category = feature_column.categorical_column_with_vocabulary_list(cat, list(df[cat].unique()))
            feature_columns.append(feature_column.embedding_column(category, dimension=8))

        feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

        return feature_layer
