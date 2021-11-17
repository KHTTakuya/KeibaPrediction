# 競馬予想モデル
pythonを用いて作成しました。

主に使用したフレームワーク

「pandas, numpy, sklearn, tensorflow, lightGBM」

# 使用方法
仮想環境orAnaconda起動後本モデルで使用するモジュールをインストールする。
```
pip intall -r requirements.txt
```

次に以下2つのファイルを作成すること
```
Keiba/datafile/
Keiba/datafile/pred_data/
```
1つ目のファイルはデータを格納するファイル、
もう1つは前処理後のファイルを格納するファイルである。
別のところに保管したい場合はファイルパスを変更の上別のファイル名を使用すること。


インストール後データを格納する。格納先は
```
Keiba/datafile/
```
直下にデータを格納する。データ名は「main.csv」とするが変更する際は
setup.pyからmain_dataのパスを書き換えること。なお本データはJRA-VANからダウンロードしているが、スクレイピングなど別の方法で得る方法でも可。スクレイピングをする際は著作権や使用方法をまもること。
本モデルで使用するデータはブログに記載してあります。

[競馬予測で使用する際のCSVデータ](https://kashiwapro.hatenablog.com/entry/2021/10/29/162155)

準備が完了したら
```
python setup.py
```
で起動します。

予想終了後「main_ans.csv」が出力されます。

# 予想方法
「main_ans.csv」を開く。

開いたら、gbm_pred, tf_pred,　flagのカラムを参照する。

flagはgbm_pred, tf_predを参照し、3着以内の確率が50％以上の物を0-1フラグで出力しています。

gbm_predはLightGBMモデルで予想した出力結果を返しています。

tf_predはTensorflowモデルで予想した出力結果を返しています。

どちらを参照して予想しても構いませんが、最初にflagを確認してその後gbm_flag、tf_pred
を確認する方法をおすすめしております。

# 免責条項
・本モデルを通して発生した損失や損害については一切責任を負うことができません。予めご了承ください。
ご自身の判断で慎重に馬券の購入をお願いいたします。

・本モデルの情報の完全性・正確性・有用性等についていかなる保証も致しません。


# リンク
ブログ ：https://kashiwapro.hatenablog.com/

Qiita ：https://qiita.com/KHTTakuya/items/35ea5e710f0fb3aa86e4

