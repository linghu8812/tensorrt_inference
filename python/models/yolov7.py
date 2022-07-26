from .YOLO import YOLO


class Yolov7(YOLO):
    def __init__(self, config):
        super().__init__(config)
