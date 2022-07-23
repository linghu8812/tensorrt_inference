#ifndef TENSORRT_INFERENCE_SCALED_YOLOV4_H
#define TENSORRT_INFERENCE_SCALED_YOLOV4_H

#include "YOLO.h"

class ScaledYOLOv4 : public YOLO {
public:
    explicit ScaledYOLOv4(const YAML::Node &config);
};

#endif //TENSORRT_INFERENCE_SCALED_YOLOV4_H
