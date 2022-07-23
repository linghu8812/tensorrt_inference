#ifndef TENSORRT_INFERENCE_MINIFASNET_H
#define TENSORRT_INFERENCE_MINIFASNET_H

#include "classification.h"

class MiniFASNet : public Classification {
public:
    explicit MiniFASNet(const YAML::Node &config);
};

#endif //TENSORRT_INFERENCE_MINIFASNET_H
