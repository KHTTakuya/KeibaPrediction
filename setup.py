import pandas as pd
import time
import warnings

from Keiba.dataprocess import KeibaProcessing
from Keiba.models import KeibaPrediction


"""
Memo:
numpy配列を加工すると処理が急激に重くなるので(16GB一杯になる)、解決できるまでは主成分分析・クラスター分析
などといったreturnがnumpy配列を返す分析手法であるものは使用不可とする。解決が出来次第処理に追加する。
※現状出ている解決策
・メモリを増やす。
・AWSなどを利用する。
"""


def create_keiba_prediction(csv_data, df_flag=True):
    # csvデータをmodelに読み込ませるように基礎加工する。
    if df_flag:
        set_data = KeibaProcessing(csv_data)
        df_main = set_data.create_dataframe()
        df_main.to_csv('Keiba/datafile/pred_data/csvdataframe.csv', encoding='utf_8_sig', index=False)
        df = pd.read_csv('Keiba/datafile/pred_data/csvdataframe.csv')
    else:
        set_data = KeibaProcessing(csv_data)
        df = pd.read_csv('Keiba/datafile/pred_data/csvdataframe.csv')

    # dfデータをLightGBM,tensorflow・logistics用に加工する
    df_gbm = set_data.data_feature_and_formating(df)
    df_logi_tf = set_data.data_feature_and_formating(df, gbmflag=False)

    # return df_logi_tf

    # df_layerは特徴量をtensorflow用に加工する。
    df_layer = set_data.df_to_tfdata(df_logi_tf)

    # csvデータをモデルに読み込ませる。
    pred_gbm = KeibaPrediction(df_gbm)
    pred = KeibaPrediction(df_logi_tf)

    # モデルを使って予測する。
    gbm = pred_gbm.gbm_params_keiba()
    tenflow = pred.tensorflow_models(df_layer)

    # 予想したものを組み合わせて出力する。
    df = pred.model_concatenation(gbm, tenflow)

    return df


if __name__ == '__main__':
    ###############################
    # メモ
    # 11.13予想の的中(1着,2着のみ)
    #
    ###############################
    warnings.simplefilter('ignore')
    # 2013年～収集開始日
    main_data = 'Keiba/datafile/main.csv'
    # 処理開始
    start = time.time()
    # 処理内容
    prediction = create_keiba_prediction(main_data, df_flag=False)
    prediction.to_csv('main_ans00001.csv', encoding='utf_8_sig')
    # 処理終了
    process_time = time.time() - start
    print('実行時間は：{} でした。'.format(process_time))  # 実行時間は：3098.3596515655518 でした。