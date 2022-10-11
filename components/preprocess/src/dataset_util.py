"""dataset utils"""
from pathlib import Path
from typing import List

from PIL import Image

from src.ann_fmt import Label, Object, load_annotation_json, CompeDataset
from src.yolo_fmt import (
    YoloObject,
    YoloLabel,
    load_yolo_format,
    save_yolo_format_txt,
)


CLASS_ID = {"car": 0}


def yolo_score_filter(
    source_dirpath: Path, output_dirpath: Path, score_threshold: float = 0.50
) -> None:
    """YOLOフォーマットスコアフィルタリング

    Args:
        label_dirpath (Path): 元データラベルパス
        output_path (Path): フィルタリングラベルパス
        score_threshold (float, optional): スコア閾値. Defaults to 0.50.
    """
    label_list = list(source_dirpath.glob("*.txt"))

    save_label_datalist = []

    for label_path in label_list:
        yolo_label = load_yolo_format(label_path)

        filtered_obj_list = []
        for obj in yolo_label.objects:
            if obj.score < score_threshold:
                continue
            filtered_obj_list.append(obj)

        save_label_datalist.append(
            YoloLabel(yolo_label.image_name, filtered_obj_list)
        )

    save_yolo_format_txt(save_label_datalist, output_dirpath)


def yolo_to_ann(
    label_dirpath: Path, image_dirpath: Path, is_score: bool = False
) -> CompeDataset:
    """コンペデータセットに変換

    Args:
        label_dirpath (Path): yoloラベルパス
        image_dirpath (Path): 画像ファイルパス
        is_score (bool, optional): スコア出力するかどうか. Defaults to False.

    Returns:
        CompeDataset: コンペデータセット
    """
    image_filepath_list = list(image_dirpath.glob("*.tif"))
    label_list = []

    for _, image_path in enumerate(image_filepath_list):

        img = Image.open(image_path)
        img_width, img_height = img.size

        label_filepath = label_dirpath.joinpath(image_path.stem + ".txt")
        # yoloラベルデータが存在しない場合、空のオブジェクトリストを生成
        if not label_filepath.exists():
            label_list.append(Label(image_path.name, []))
            continue

        # yoloラベルデータが存在
        yolo_label = load_yolo_format(label_filepath)
        obj_list = []
        for obj in yolo_label.objects:
            class_name = [k for k, v in CLASS_ID.items() if v == obj.class_][0]
            xyxy = obj.xyxy()
            score = obj.score if is_score else None
            obj_list.append(
                Object(
                    class_=class_name,
                    lefttop_x=xyxy[0] * img_width,
                    lefttop_y=xyxy[1] * img_height,
                    rightbottom_x=xyxy[2] * img_width,
                    rightbottom_y=xyxy[3] * img_height,
                    score=score,
                )
            )
        label_list.append(Label(image_path.name, obj_list))

    return CompeDataset(label_list)


def ann_to_yolo(label_filepath: Path, image_dirpath: Path) -> List[YoloLabel]:
    """YOLOフォーマットに変換

    Args:
        label_filepath (Path): ラベルファイルパス
        image_dirpath (Path): 画像ファイルパス

    Returns:
        List[YoloLabel]: YOLOアノテーションデータ
    """
    ann_data = load_annotation_json(label_filepath)

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
