"""yoloフォーマット"""
from pathlib import Path
from typing import Optional, List


class YoloObject:

    """YOLOフォーマットオブジェクト"""

    def __init__(
        self,
        class_: int,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        score: Optional[float] = None,
    ) -> None:
        """コンストラクタ

        Args:
            class_ (int): クラスID
            x_center (float): BBOX重心x座標
            y_center (float): BBOX重心y座標
            width (float): BBOX幅
            height (float): BBOX高さ
        """
        self.class_ = class_
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.score = score

    def xyxy(self) -> List[float]:
        """xyxy座標

        Returns:
            List[float]: _description_
        """
        return [
            self.x_center - self.width / 2,
            self.y_center - self.height / 2,
            self.x_center + self.width / 2,
            self.y_center + self.height / 2,
        ]


class YoloLabel:

    """YOLOラベルフォーマット"""

    def __init__(self, image_name: str, objects: List[YoloObject]) -> None:
        """コンストラクタ

        Args:
            image_name (str): ファイル名
            objects (List[YoloObject]): BBOXオブジェクトリスト
        """
        self.image_name = image_name
        self.objects = objects

    def to_text(self) -> List[str]:
        """ラベルテキストフォーマットへ変換

        Returns:
            List[str]: _description_
        """
        str_list = []
        for obj in self.objects:
            txt = " ".join(
                [
                    f"{obj.class_}",
                    f"{obj.x_center:.6f}",
                    f"{obj.y_center:.6f}",
                    f"{obj.width:.6f}",
                    f"{obj.height:.6f}",
                ]
            )
            if obj.score is not None:
                txt = txt + f" {obj.score}"
            str_list.append(txt)

        return str_list


def load_yolo_format(filepath: Path) -> YoloLabel:
    """yoloフォーマット読み込み

    Args:
        filepath (Path): ファイルパス

    Returns:
        YoloLabel: ラベルデータ
    """
    with filepath.open(mode="r") as f:
        obj_list = []
        for line in f:
            txt = line.rstrip("\n").split(sep=" ")
            if len(txt) == 5:
                obj = YoloObject(
                    class_=int(txt[0]),
                    x_center=float(txt[1]),
                    y_center=float(txt[2]),
                    width=float(txt[3]),
                    height=float(txt[4]),
                )
            else:
                obj = YoloObject(
                    class_=int(txt[0]),
                    x_center=float(txt[1]),
                    y_center=float(txt[2]),
                    width=float(txt[3]),
                    height=float(txt[4]),
                    score=float(txt[5]),
                )

            obj_list.append(obj)

    filename = filepath.stem + ".tif"
    return YoloLabel(image_name=filename, objects=obj_list)


def save_yolo_format_txt(
    label_data: List[YoloLabel], outpur_dirpath: Path
) -> None:
    """yoloフォーマット保存

    Args:
        label_data (List[YoloLabel]): ラベルデータ
        outpur_dirpath (Path): ディレクトリパス
    """
    if not outpur_dirpath.exists():
        outpur_dirpath.mkdir(parents=True, exist_ok=True)

    for d in label_data:
        filepath = outpur_dirpath.joinpath(Path(d.image_name).stem + ".txt")
        with filepath.open(mode="w") as f:
            for line in d.to_text():
                f.write(line + "\n")
