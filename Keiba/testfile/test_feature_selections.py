from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import (VarianceThreshold, SelectKBest,
                                       mutual_info_classif, SequentialFeatureSelector, RFE, RFECV)
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.ensemble import RandomForestClassifier
from Keiba.dataprocess import KeibaProcessing


class ChooseFeatureFilterMethod:

    def __init__(self, data):
        # data = csv file
        df = pd.read_csv(data)
        get_form = KeibaProcessing(data)

        df = get_form.data_feature_and_formating(df, gbmflag=False)
        df['days'] = pd.to_datetime(df['days'])
        df = df[df['days'] >= datetime(2021, 11, 20)]

        new_df = df.loc[:, "pop":]
        new_df = new_df.drop('father_legtype', axis=1)
        new_df = pd.DataFrame(new_df)

        new_df = new_df.dropna(how='any')

        self.new_df = new_df

    def filter_method(self):
        new_df = self.new_df
        new_df = new_df.drop('flag', axis=1)

        sel = VarianceThreshold()
        sel.fit_transform(new_df)

        x_new = pd.DataFrame(sel.fit_transform(new_df))
        result = pd.DataFrame(sel.get_support(), index=new_df.columns.values, columns=['False: dropped'])
        result['variance'] = sel.variances_

        Before = f'Before Feature Selection: {new_df.shape}'
        After = f'After Feature Selection: {x_new.shape}'
        variance = result[result['variance'] < 0.001]
        return result.to_csv('main_tst_filter.csv', encoding='utf_8-sig'), Before, After

    def info_selection(self, plots=True):
        new_df = self.new_df

        selector = SelectKBest(mutual_info_classif, k=2)
        X_new2 = pd.DataFrame(selector.fit_transform(new_df, new_df['flag']),
                              columns=new_df.columns.values[selector.get_support()])
        result = pd.DataFrame(selector.get_support(), index=new_df.columns.values, columns=['False: dropped'])
        pd.options.display.float_format = None
        result['score'] = selector.scores_

        if plots:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))
            fig.subplots_adjust(wspace=0.3, hspace=0.2)

            result.sort_values('score', ascending=True, inplace=True)
            result['score'].plot.barh(ax=axes[0], stacked=True, y=[0, 1])
            axes[1].axis('tight')
            axes[1].axis('off')
            axes[1].table(cellText=result.round(3).values,  # roundしないと表が小さすぎる
                          colLabels=result.columns,
                          rowLabels=result.index, loc='center')
            return plt.show()

        else:
            return result.sort_values('score')


class ChooseFeatureWrapperMethod:

    def __init__(self, data, f_select=80):
        df = pd.read_csv(data)
        get_form = KeibaProcessing(data)

        df = get_form.data_feature_and_formating(df, gbmflag=False)
        df['days'] = pd.to_datetime(df['days'])
        df = df[df['days'] >= datetime(2021, 11, 20)]

        new_df = df.loc[:, "pop":]
        new_df = new_df.drop('father_legtype', axis=1)
        new_df = pd.DataFrame(new_df)

        new_df = new_df.dropna(how='any')

        sel = VarianceThreshold()
        X = pd.DataFrame(sel.fit_transform(new_df), columns=new_df.columns.values[sel.get_support()])

        self.df = new_df
        self.rf = RandomForestClassifier()
        self.X = X
        self.f_select = f_select

    def wrapper_method_forward_selection(self):
        df = self.df

        selector = SequentialFeatureSelector(self.rf, n_features_to_select=self.f_select, cv=5)

        X_new = pd.DataFrame(selector.fit_transform(self.X, df['flag']),
                             columns=self.X.columns.values[selector.get_support()])
        result = pd.DataFrame(selector.get_support(), index=self.X.columns.values, columns=['False: dropped'])

        return result.to_csv('main_tst.csv', encoding='utf_8-sig')

    def wrapper_method_backward_selection(self):
        df = self.df

        selector = SequentialFeatureSelector(self.rf, n_features_to_select=self.f_select, direction='backward', cv=5)

        X_new = pd.DataFrame(selector.fit_transform(self.X, df['flag']),
                             columns=self.X.columns.values[selector.get_support()])
        result = pd.DataFrame(selector.get_support(), index=self.X.columns.values, columns=['False: dropped'])

        return result.to_csv('main_test_backward.csv', encoding='utf_8_sig')

    def wrapper_method_ref(self, plot_flag=False):
        df = self.df

        selector = RFECV(self.rf, min_features_to_select=self.f_select, cv=10)

        X_new = pd.DataFrame(selector.fit_transform(self.X, df['flag']),
                             columns=self.X.columns.values[selector.get_support()])
        result = pd.DataFrame(selector.get_support(), index=self.X.columns.values, columns=['False: dropped'])
        result['ranking'] = selector.ranking_

        if plot_flag:
            plt.figure()
            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score (nb of correct classifications)")
            plt.plot(range(self.f_select,
                           len(selector.grid_scores_) + self.f_select),
                     selector.grid_scores_)
            return plt.show()

        return result.to_csv('tet_ref.csv', encoding='utf_8_sig')

    def wrapper_method_efs(self):
        df = self.df

        efs1 = EFS(self.rf, min_features=9, max_features=self.f_select)
        efs1 = efs1.fit(self.X, df['flag'])

        df_main = pd.DataFrame.from_dict(efs1.get_metric_dict()).T
        df_main.sort_values('avg_score', inplace=True, ascending=False)

        return df_main.to_csv('tet_efs.csv', encoding='utf_8_sig')
