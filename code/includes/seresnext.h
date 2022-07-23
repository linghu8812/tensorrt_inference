#ifndef TENSORRT_INFERENCE_SERESNEXT_H
#define TENSORRT_INFERENCE_SERESNEXT_H

#include "classification.h"

class SEResNeXt : public Classification {
public:
    explicit SEResNeXt(const YAML::Node &config);
};

#endif //TENSORRT_INFERENCE_SERESNEXT_H
