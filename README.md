# Vehicle Detection in Multi-Resolution Images 
## 本リポジトリについて
### ディレクトリ構成
```
multi-resolution-vehicle-detection
├── README.md           # 本書
└── components
    ├── preprocess      # 1.前処理（データセット生成）
    ├── train           # 2.モデル学習
    └── predict         # 3.推論
```


## 前処理（データセット生成）
## 概要
- コンペフォーマットのアノテーションデータ`train.json`をYOLOフォーマットに変換
- モデル学習のため、データセットをtrain/valに分割

## 実行環境
- M1 Mac
- MacOS Monterey
- python >= 3.10
- poetry >= 1.1

## 実行方法
- 詳細は[conponents/preprocess/ReadMe.md](components/preprocess/Readme.md)を参照

## モデル学習
### 概要
#### モデル学習
  - 前処理で作成したデータセットファイルを参照して、モデル学習
  - モデルはオープンソースのYOLOv5を利用
    - Github: https://github.com/ultralytics/yolov5
    - Lisence: [GNU General Public License v3.0](https://github.com/ultralytics/yolov5/blob/master/LICENSE)
#### OC-Cost最適化のための検出スコア閾値設定
  - validationデータを使って、OC-Costが最小となる検出スコア閾値を探索

## 実行環境
- Google Colabratory上でNotebookを実行
- ランタイムはGPUを利用
  - GPU: NVIDIA Tesla T4
  - CUDA: 11.2
  - Python: 3.7
- train/valデータセットはGoogle Driveへアップロードしておき、実行環境とマウント
### 実行方法
- 詳細は[conponents/train/ReadMe.md](components/train/Readme.md)を参照

## モデル評価
### 概要
- 学習済みモデルと検出スコア探索結果をもとに評価データセットへ推論処理、複数閾値パターンの予測結果を出力

## 実行環境
- 学習プロセスと同じくGoogle Colabratoryを利用
## 実行方法
- 詳細は[conponents/predict/ReadMe.md](components/predict/Readme.md)を参照
