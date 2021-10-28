# numdatas：数字のカラムを認識する用。l
num_datas = ['horsenum', 'pre_pop', 'pre_odds', 'pre_rank3', 'pre_rank4', 'pre_3ftime', 'pre_result', 'jockey_tyukyou',
             'jockey_nakayama', 'jockey_kyoto', 'jockey_hakodate', 'jockey_kokura', 'jockey_niigata', 'jockey_sapporo',
             'jockey_tokyo', 'jockey_fukushima', 'jockey_hannshin', 'father_tyukyou', 'father_nakayama', 'father_kyoto',
             'father_hakodate', 'father_kokura', 'father_niigata', 'father_sapporo', 'father_tokyo', 'father_fukushima',
             'father_hannshin', 'father_mile', 'father_tyukyori', 'father_tankyori', 'father_tyoukyori', 'father_d',
             'father_t', 'father_cf', 'father_cfy','father_cy', 'father_cg', 'father_co', 'father_3ftime', 'fathertype_tyukyou',
             'fathertype_nakayama', 'fathertype_kyoto', 'fathertype_hakodate', 'fathertype_kokura',
             'fathertype_niigata',
             'fathertype_sapporo', 'fathertype_tokyo', 'fathertype_fukushima', 'fathertype_hannshin', 'fathertype_mile',
             'fathertype_tyukyori', 'fathertype_tankyori', 'fathertype_tyoukyori', 'fathertype_d', 'fathertype_t',
             'fathertype_cf', 'fathertype_cfy', 'fathertype_cy', 'fathertype_cg', 'fathertype_co', 'fathertype_3ftime', 'place_odds',
             'place_pop', 'place_rank3', 'place_rank4', 'place_3ftime', 'distance_odds', 'distance_pop',
             'distance_time',
             'distance_rank3', 'distance_rank4', 'distance_3ftime', 'odds_hi', 're_odds_hi', 're_3_to_4time',
             'father_3f_to_my', 'fathertype_3f_to_my', 're_pop_now_pop', 're_odds_now_odds', 're_result_to_pop'
             ]

# 名前を変える前のリスト
re_rename_list = ["jockey_中京", "jockey_中山", "jockey_京都", "jockey_函館", "jockey_小倉", "jockey_新潟", "jockey_札幌",# 7
                  "jockey_東京", "jockey_福島", "jockey_阪神", "father_中京", "father_中山", "father_京都", "father_函館",# 7
                  "father_小倉", "father_新潟", "father_札幌", "father_東京", "father_福島", "father_阪神", "father_マイル", # 7
                  "father_中距離", "father_短距離", "father_長距離", "father_ダ", "father_芝", "father_不", "father_不良", # 7
                  "father_稍", "father_良", "father_重", "fathertype_中京", "fathertype_中山", "fathertype_京都", #6
                  "fathertype_函館", "fathertype_小倉", "fathertype_新潟", "fathertype_札幌", "fathertype_東京", #5
                  "fathertype_福島", "fathertype_阪神", "fathertype_マイル", "fathertype_中距離", "fathertype_短距離", #5
                  "fathertype_長距離", "fathertype_ダ", "fathertype_芝", "fathertype_不", "fathertype_不良", "fathertype_稍", #6
                  "fathertype_良", "fathertype_重"]

# 名前を変えた名前のリスト
rename_list = ['jockey_tyukyou', 'jockey_nakayama', 'jockey_kyoto', 'jockey_hakodate', 'jockey_kokura', 'jockey_niigata', 'jockey_sapporo', #7
               'jockey_tokyo', 'jockey_fukushima', 'jockey_hannshin', 'father_tyukyou', 'father_nakayama', 'father_kyoto','father_hakodate', #7
               'father_kokura', 'father_niigata', 'father_sapporo', 'father_tokyo', 'father_fukushima', 'father_hannshin', 'father_mile', # 7
               'father_tyukyori', 'father_tankyori', 'father_tyoukyori', 'father_d', 'father_t', 'father_cf', 'father_cfy', # 7
               'father_cy', 'father_cg', 'father_co','fathertype_tyukyou', 'fathertype_nakayama', 'fathertype_kyoto', # 6
               'fathertype_hakodate', 'fathertype_kokura', 'fathertype_niigata', 'fathertype_sapporo', 'fathertype_tokyo', #5
               'fathertype_fukushima', 'fathertype_hannshin', 'fathertype_mile', 'fathertype_tyukyori', 'fathertype_tankyori', #5
               'fathertype_tyoukyori', 'fathertype_d', 'fathertype_t', 'fathertype_cf', 'fathertype_cfy', 'fathertype_cy', #6
               'fathertype_cg', 'fathertype_co']
