import warnings
import pandas as pd

from Keiba.testfile.test_dataprocess import TestDataProcess
from Keiba.testfile.test_models import TestPredictionModelGBM, TestPredictModelTF, TestMergeModelDataToCsv


def combine(data):
    start = TestDataProcess(data)
    df = start.add_feature_formatting_process(switch=False)
    df_new = df.copy()
    df_tnf = df.copy()
    gbm_model = TestPredictionModelGBM(df).model()
    tnf_model = TestPredictModelTF(df_tnf).models()
    ans_data1 = TestMergeModelDataToCsv(df_new, gbm_model, tnf_model)

    return ans_data1.merged_data()


# test用特徴量の増減などはここで処理をする。
if __name__ == '__main__':
    warnings.simplefilter('ignore')
    main_data = 'Keiba/datafile/main.csv'
    start = TestDataProcess(main_data)
    df = start.add_feature_formatting_process()
    print(df)
    # df = combine(main_data)
    # print(df)
