import time
import warnings

from Keiba.dataprocess import KeibaProcessing
from Keiba.models import KeibaPrediction


def create_keiba_prediction(csv_data):
    # 　csvデータをmodelに読み込ませるように基礎加工する。
    set_data = KeibaProcessing(csv_data)
    df = set_data.create_dataframe()

    # dfデータをLightGBM,tensorflow・logistics用に加工する
    df_gbm = set_data.data_feature_and_formating(df)
    df_logi_tf = set_data.data_feature_and_formating(df, gbmflag=False)

    # df_layerは特徴量をtensorflow用に加工する。
    df_layer = set_data.df_to_tfdata(df_logi_tf)

    # csvデータをモデルに読み込ませる。
    pred_gbm = KeibaPrediction(df_gbm)
    pred = KeibaPrediction(df_logi_tf)

    # モデルを使って予測する。
    gbm = pred_gbm.gbm_params_keiba()
    tenflow = pred.tensorflow_models(df_layer)
    log = pred.logistic_model()

    # 予想したものを組み合わせて出力する。
    df = pred.model_concatenation(gbm, tenflow, log)

    return df


if __name__ == '__main__':
    warnings.simplefilter('ignore')

    main_data = 'Keiba/datafile/main.csv'
    # 処理開始
    start = time.time()
    # 処理内容
    prediction = create_keiba_prediction(main_data)
    prediction.to_csv('main_ans.csv', encoding='utf_8_sig')
    # 処理終了
    process_time = time.time() - start
    print('実行時間は：{} でした。'.format(process_time))  # 実行時間は：3098.3596515655518 でした。
