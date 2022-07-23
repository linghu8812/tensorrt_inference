#ifndef TENSORRT_INFERENCE_ALEXNET_H
#define TENSORRT_INFERENCE_ALEXNET_H

#include "classification.h"

class AlexNet : public Classification {
public:
    explicit AlexNet(const YAML::Node &config);
};

#endif //TENSORRT_INFERENCE_ALEXNET_H
