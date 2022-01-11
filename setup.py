import time
import warnings

from Keiba.dataprocess import DataProcess
from Keiba.models import PredictionModelGBM, PredictModelTF, MergeModelDataToCsv


def create_keiba_prediction(csv_data, flag=True):
    """
    :param csv_data: 初期設定：'Keiba/datafile/main.csv'
    :param flag: Trueは前処理からスタート、Falseは予想からスタート
    :return: csv(ans.csv, encoding='utf_8_sig')
    """
    # データ前処理
    df_start = DataProcess(csv_data=csv_data)
    df = df_start.add_feature_formatting_process(switch=flag)
    # pandas dataframeをコピーする。
    df_merge = df.copy()
    df_tnf = df.copy()
    # モデルにいれて予想、結果を返す。
    gbm_model = PredictionModelGBM(df).model()
    tnf_model = PredictModelTF(df_tnf).models()
    ans_data1 = MergeModelDataToCsv(df_merge, gbm_model, tnf_model)

    return ans_data1.merged_data()


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    # 2013年～収集開始日
    main_data = 'Keiba/datafile/main.csv'
    # 処理開始
    start = time.time()
    # 処理内容(flag=True or Falseに書き換えは可)
    prediction = create_keiba_prediction(main_data, flag=False)
    print(prediction)
    # 処理終了
    process_time = time.time() - start
    print('実行時間は：{} でした。'.format(process_time))
