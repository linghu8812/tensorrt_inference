#ifndef TENSORRT_INFERENCE_LENET_H
#define TENSORRT_INFERENCE_LENET_H

#include "classification.h"

class LeNet : public Classification {
public:
    explicit LeNet(const YAML::Node &config);
};

#endif //TENSORRT_INFERENCE_LENET_H
