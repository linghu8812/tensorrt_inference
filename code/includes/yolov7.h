#ifndef TENSORRT_INFERENCE_YOLOV7_H
#define TENSORRT_INFERENCE_YOLOV7_H

#include "YOLO.h"

class yolov7 : public YOLO {
public:
    explicit yolov7(const YAML::Node &config);
};

#endif //TENSORRT_INFERENCE_YOLOV7_H
