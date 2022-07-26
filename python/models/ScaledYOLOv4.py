from .YOLO import YOLO


class ScaledYOLOv4(YOLO):
    def __init__(self, config):
        super().__init__(config)
