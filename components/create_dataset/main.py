"""to yolov5 format"""
from pathlib import Path
import shutil

import fire
from PIL import Image
from sklearn.model_selection import train_test_split

from src.ann_fmt import Label, load_annotation_json
from src.yolo_fmt import YoloObject, YoloLabel

CLASS_ID = {"car": 0}
RANDOM_SEED = 42


def to_yolo_format(
    ann_data: list[Label], image_dirpath: Path
) -> list[YoloLabel]:
    """YOLOフォーマットに変換

    Args:
        ann_data (list[Label]): コンペアノテーションデータ
        image_dirpath (Path): 画像ファイルパス

    Returns:
        list[YoloLabel]: YOLOアノテーションデータ
    """
    label_list = []

    for _, ann in enumerate(ann_data):

        img_filepath = image_dirpath.joinpath(ann.name)
        img = Image.open(img_filepath)
        img_width, img_height = img.size

        obj_list = []

        for _, obj in enumerate(ann.objects):
            cx, cy = obj.bbox_centor()
            center_x = cx / img_width
            center_y = cy / img_height
            width = obj.width / img_width
            height = obj.height / img_height
            obj_list.append(
                YoloObject(
                    CLASS_ID[obj.class_], center_x, center_y, width, height
                )
            )

        label_list.append(YoloLabel(ann.name, obj_list))

    return label_list


def split_dataset(
    yolo_labels: list[YoloLabel], val_size: float = 0.1
) -> tuple[list[YoloLabel], list[YoloLabel]]:
    """データセット分割

    Args:
        yolo_labels (list[YoloLabel]): データセット
        val_size (float): validationデータセット割合（デフォルト: 0.1）

    Returns:
        tuple[list[YoloLabel], list[YoloLabel]]: train/valラベルデータセット
    """
    train, val = train_test_split(
        yolo_labels,
        test_size=val_size,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    return train, val


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

    # データセット取得
    json_data = load_annotation_json(Path(label_filepath))

    # yolo変換
    yolo_data = to_yolo_format(json_data, Path(image_dirpath))

    # データセット分割
    train, val = split_dataset(yolo_data, val_size)

    # データセット出力
    output_dataset(
        train, Path(image_dirpath), Path(output_dirpath).joinpath("train")
    )
    output_dataset(
        val, Path(image_dirpath), Path(output_dirpath).joinpath("val")
    )


if __name__ == "__main__":

    fire.Fire(main)
