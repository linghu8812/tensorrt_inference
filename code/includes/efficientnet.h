#ifndef TENSORRT_INFERENCE_EFFICIENTNET_H
#define TENSORRT_INFERENCE_EFFICIENTNET_H

#include "classification.h"

class EfficientNet : public Classification {
public:
    explicit EfficientNet(const YAML::Node &config);
};

#endif //TENSORRT_INFERENCE_EFFICIENTNET_H
