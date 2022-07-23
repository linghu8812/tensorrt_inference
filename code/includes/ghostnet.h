#ifndef TENSORRT_INFERENCE_GHOSTNET_H
#define TENSORRT_INFERENCE_GHOSTNET_H

#include "classification.h"

class GhostNet : public Classification {
public:
    explicit GhostNet(const YAML::Node &config);
};

#endif //TENSORRT_INFERENCE_GHOSTNET_H
