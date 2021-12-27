import warnings
import pandas as pd
from Keiba.models import PredictModelTF

# test用特徴量の増減などはここで処理をする。
if __name__ == '__main__':
    warnings.simplefilter('ignore')

    get_data = 'Keiba/datafile/pred_data/csvdataframe.csv'

    df_test = pd.read_csv(get_data)

    # get_filter = ChooseFeatureFilterMethod(get_data)
    # get_wrapper = ChooseFeatureWrapperMethod(get_data)

    # tf_data = TestTFModel(get_data)
    # print(tf_data.models())

    df_main = PredictModelTF(get_data)
    print(df_main.models())