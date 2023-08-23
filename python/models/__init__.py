import yaml
from .ScaledYOLOv4 import ScaledYOLOv4
from .YOLOv4 import Yolov4
from .yolor import Yolor
from .yolov5 import Yolov5
from .YOLOv6 import YOLOv6
from .yolov7 import Yolov7
from .yolov8 import Yolov8

__factory = {
    'ScaledYOLOv4': ScaledYOLOv4,
    'Yolov4': Yolov4,
    'yolor': Yolor,
    'yolov5': Yolov5,
    'YOLOv6': YOLOv6,
    'yolov7': Yolov7,
    'yolov8': Yolov8,
}


def build_model(opt):
    """
    Create a backbone model.
    Parameters
    ----------
    name : str
        The backbone name.
    pretrained : str
        ImageNet pretrained.
    """
    if opt.arch not in __factory:
        raise KeyError("Unknown model:", opt.arch)
    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)
    return __factory[opt.arch](config[opt.arch])
