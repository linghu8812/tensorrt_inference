#ifndef TENSORRT_INFERENCE_YOLOR_H
#define TENSORRT_INFERENCE_YOLOR_H

#include "YOLO.h"

class yolor : public YOLO {
public:
    explicit yolor(const YAML::Node &config);
};

#endif //TENSORRT_INFERENCE_YOLOR_H
