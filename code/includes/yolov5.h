#ifndef TENSORRT_INFERENCE_YOLOV5_H
#define TENSORRT_INFERENCE_YOLOV5_H

#include "YOLO.h"

class YOLOv5 : public YOLO {
public:
    explicit YOLOv5(const YAML::Node &config);
};

class YOLOv5_cls :public Classification {
public:
    explicit YOLOv5_cls(const YAML::Node &config);
};

#endif //TENSORRT_INFERENCE_YOLOV5_H
