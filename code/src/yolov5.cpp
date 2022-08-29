#include "yolov5.h"

YOLOv5::YOLOv5(const YAML::Node &config) : YOLO(config) {}

YOLOv5_cls::YOLOv5_cls(const YAML::Node &config) : Classification(config) {}
