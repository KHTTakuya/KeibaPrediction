import warnings
import pandas as pd

from Keiba.dataprocess import KeibaProcessing
from Keiba.models import KeibaPrediction

if __name__ == '__main__':
    warnings.simplefilter('ignore')

    data = 'Keiba/datafile/main.csv'
    predict_data = 'Keiba/datafile/pred.csv'
    # df = KeibaProcessing(csv_data=data)
    # df = df.create_df().to_csv('Keiba/datafile/pred_data/ans.csv', encoding='utf_8_sig')
    dataframe = pd.read_csv('Keiba/datafile/pred_data/ans.csv')
    pred = KeibaPrediction(dataframe)
    print(pred.gbm_params_keiba().to_csv('Keiba/datafile/pred_data/prediction.csv', encoding='utf_8_sig'))