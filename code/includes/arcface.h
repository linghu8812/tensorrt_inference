#ifndef TENSORRT_INFERENCE_ARCFACE_H
#define TENSORRT_INFERENCE_ARCFACE_H

#include "feature.h"

class ArcFace : public Feature {
public:
    explicit ArcFace(const YAML::Node &config);
};

#endif //TENSORRT_INFERENCE_ARCFACE_H
