"""Vehicle Detection in Multi-Resolution Imagesデータセット"""
from pathlib import Path
import json
from typing import Optional


class Object:

    """オブジェクトクラス"""

    def __init__(
        self,
        class_: str,
        lefttop_x: float,
        lefttop_y: float,
        rightbottom_x: float,
        rightbottom_y: float,
        score: Optional[float] = None,
    ) -> None:
        """コンストラクタ

        Args:
            class_ (str): クラス名
            lefttop_x (float): BBOX左上x座標
            lefttop_y (float): BBOX左上y座標
            rightbottom_x (float): BBOX右下x座標
            rightbottom_y (float): BBOX右下y座標
        """
        self.class_ = class_
        self.lefttop_x = lefttop_x
        self.lefttop_y = lefttop_y
        self.rightbottom_x = rightbottom_x
        self.rightbottom_y = rightbottom_y
        self.score = score

    def to_dataset_format(self) -> dict:
        """データセット型へ変換

        Returns:
            dict: _description_
        """
        data = {
            "class": self.class_,
            "lefttop_x": self.lefttop_x,
            "lefttop_y": self.lefttop_y,
            "rightbottom_x": self.rightbottom_x,
            "rightbottom_y": self.rightbottom_y,
        }
        if self.score is not None:
            data["precision"] = self.score
        return data

    @property
    def width(self) -> float:
        """bbox幅"""
        return self.rightbottom_x - self.lefttop_x

    @property
    def height(self) -> float:
        """bbox高さ"""
        return self.rightbottom_y - self.lefttop_y

    def bbox_xyxy(self) -> list[float]:
        """xyxy形式

        Returns:
            list[float]: _description_
        """
        return [
            self.lefttop_x,
            self.lefttop_y,
            self.rightbottom_x,
            self.rightbottom_y,
        ]

    def bbox_centor(self) -> tuple[float, float]:
        """bbox重心座標"""
        return (
            (self.lefttop_x + self.rightbottom_x) / 2,
            (self.lefttop_y + self.rightbottom_y) / 2,
        )


class Label:

    """ラベルクラス"""

    def __init__(self, name: str, objects: list[Object]) -> None:
        """コンストラクタ

        Args:
            name (str): 画像ファイル名
            objects (list[Object]): BBOXオブジェクトリスト
        """
        self.name = name
        self.objects = objects

    def to_dataset_format(self) -> dict:
        """データセット型へ変換

        Returns:
            dict: _description_
        """
        return {
            "name": self.name,
            "annotation": [obj.to_dataset_format() for obj in self.objects],
        }


class CompeDataset:

    """データセット"""

    def __init__(self, labels: list[Label]) -> None:
        """コンストラクタ

        Args:
            labels (list[Label]): _description_
        """
        self.labels = labels

    def to_dataset_format(self) -> dict:
        """データセット型へ変換

        Returns:
            dict: _description_
        """
        return {
            "images": [label.to_dataset_format() for label in self.labels],
        }


def load_annotation_json(filepath: Path) -> list[Label]:
    """データセット読み込み

    Args:
        filepath (Path): データセットファイルパス

    Returns:
        list[Label]: データセットラベルリスト
    """
    with filepath.open(mode="r") as f:
        json_data = json.load(f)

    img_list = []
    for img in json_data["images"]:
        ann_list = []
        for ann in img["annotation"]:
            ann_list.append(
                Object(
                    ann["class"],
                    ann["lefttop_x"],
                    ann["lefttop_y"],
                    ann["rightbottom_x"],
                    ann["rightbottom_y"],
                    ann["precision"] if "precision" in ann else None,
                )
            )
        img_list.append(Label(img["name"], ann_list))

    return img_list


def save_annotation_json(label_data: list[Label], filepath: Path) -> None:
    """アノテーションファイル保存

    Args:
        label_data (list[Label]): ラベルデータ
        filepath (Path): 出力ファイルパス
    """
    images = []

    for d in label_data:
        images.append(d.to_dataset_format())
    with filepath.open(mode="w", encoding="utf-8") as f:
        json_data = {"images": images}
        json.dump(json_data, f)
