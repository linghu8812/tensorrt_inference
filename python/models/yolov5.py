from .YOLO import YOLO


class Yolov5(YOLO):
    def __init__(self, config):
        super().__init__(config)
