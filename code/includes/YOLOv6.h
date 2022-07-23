#ifndef TENSORRT_INFERENCE_YOLOV6_H
#define TENSORRT_INFERENCE_YOLOV6_H

#include "YOLO.h"

class YOLOv6 : public YOLO {
public:
    explicit YOLOv6(const YAML::Node &config);
};

#endif //TENSORRT_INFERENCE_YOLOV6_H
