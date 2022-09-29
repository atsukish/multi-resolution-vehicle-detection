"""to yolov5 format"""
from pathlib import Path
import shutil

import fire
from sklearn.model_selection import train_test_split

from src.yolo_fmt import YoloLabel
from src.dataset_util import ann_to_yolo

RANDOM_SEED = 101


def output_dataset(
    yolo_labels: list[YoloLabel], image_dir: Path, output_dir: Path
) -> None:
    """データセット出力

    Args:
        yolo_labels (list[YoloLabel]): yolo形式ラベルデータ
        image_dir (Path): 画像ディレクトリパス
        output_dir (Path): 出力ディレクトリパス
    """
    # ディレクトリ作成
    output_image_dir = output_dir.joinpath("images")
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir = output_dir.joinpath("labels")
    output_label_dir.mkdir(parents=True, exist_ok=True)

    for label in yolo_labels:
        # yoloラベル生成
        label_filepath = output_label_dir.joinpath(
            Path(label.image_name).stem + ".txt"
        )
        with label_filepath.open(mode="w") as f:
            for line in label.to_text():
                f.write(line + "\n")
        # 画像コピー
        image_filepath = image_dir.joinpath(
            Path(label.image_name).stem + ".tif"
        )
        image_outfilepath = output_image_dir.joinpath(
            Path(label.image_name).stem + ".tif"
        )
        shutil.copy(image_filepath, image_outfilepath)


def main(
    label_filepath: str,
    image_dirpath: str,
    output_dirpath: str,
    val_size: float = 0.1,
) -> None:
    """メイン

    Args:
        label_filepath (str): アノテーションjsonファイルパス
        image_dirpath (str): 画像ファイルパス
        yolo_dirpath (str): 出力yoloディレクトリパス
        val_size (float): テストデータ割合（デフォルト:0.1）

    Raises:
        FileNotFoundError: _description_
        FileNotFoundError: _description_
    """
    # データセットファイル確認
    if not Path(label_filepath).exists():
        raise FileNotFoundError
    # 画像ディレクトリ確認
    if not Path(image_dirpath).is_dir():
        raise FileNotFoundError

    # json to yoloフォーマット
    yolo_data = ann_to_yolo(Path(label_filepath), Path(image_dirpath))

    # データセット分割
    train, val = train, val = train_test_split(
        yolo_data,
        test_size=val_size,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    # データセット出力
    output_dataset(
        train, Path(image_dirpath), Path(output_dirpath).joinpath("train")
    )
    output_dataset(
        val, Path(image_dirpath), Path(output_dirpath).joinpath("val")
    )


if __name__ == "__main__":

    fire.Fire(main)
