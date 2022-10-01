# Create Dataset

## Overview

- モデル学習向けにデータセットを生成
  - コンペティションのラベルデータjsonファイルからYOLOフォーマットに変換
  - train/validationデータセットに分割

## Requierement

- MacOS Monterey
- python >= 3.10
- no GPU
- poetry >= 1.1

## Setup

- poetry

  ```shell
  $ cd components/create_dataset
  $ poetry install
  ```
- pip
  ```shell
  $ cd components/create_dataset
  $ pip install -r requirements.txt
  ```

## Create Dataset

- main.pyを実行

  ```shell
  $ poetry run python main.py \
    　--label_filepath /hoge/label/train/train.json　\
    　--image_dirpath /hoge/images/train/ \
    　--output_dirpath /hoge/yolo_dataset/ \
    　--val_size 0.1
  ```

  | #    | 引数               | 説明                             | データ型 | 必須                |
  | ---- | ------------------ | -------------------------------- | -------- | ------------------- |
  | 1    | `--label_filepath` | 変換前ラベルjsonファイルパス     | str      | ○                   |
  | 2    | `--image_dir`      | 画像ディレクトリパス             | str      | ○                   |
  | 3    | `--output_dirpath` | 出力データセットディレクトリパス | str      | ○                   |
  | 4    | `--val_size`      | validataionデータセット割合      | float    | ☓（デフォルト:0.1） |

  

- `output_dir`ディレクトリ構成
  - `train`: 学習データセット
  - `val`: 検証データセット

  ```shell
  └──output_dir
     ├── train
     │   ├── images
     │   │   ├── train_xx.tif
     │   │   ├── train_yy.tif
     │   │   ...
     │   └── labels
     │       ├── train_xx.txt
     │       ├── train_yy.txt
     │       ...
     └─── val
         ├── images
         │   ├── train_zz.tif
         │   ├── train_ii.tif
         │   ...
         └── labels
             ├── train_zz.txt
             ├── train_ii.txt
             ...
  ```