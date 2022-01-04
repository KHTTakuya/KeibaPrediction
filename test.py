import warnings
import pandas as pd

from Keiba.testfile.test_dataprocess import TestDataProcess
from Keiba.testfile.test_models import TestGBMModel
from Keiba.dataprocess import KeibaProcessing
from Keiba.models import PredictModelTF, KeibaPrediction


def combine(data):
    start = TestDataProcess(data)
    df = start.add_feature_formatting_process()
    model = TestGBMModel(df)
    pred = model.model()

    return pred


# test用特徴量の増減などはここで処理をする。
if __name__ == '__main__':
    warnings.simplefilter('ignore')

    main_data = 'Keiba/datafile/main.csv'
    df = combine(main_data)
    print(df.to_csv('test_data.csv', encoding='utf_8_sig'))
    # print(df)
