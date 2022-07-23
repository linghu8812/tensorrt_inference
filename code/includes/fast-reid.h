#ifndef TENSORRT_INFERENCE_FAST_REID_H
#define TENSORRT_INFERENCE_FAST_REID_H

#include "feature.h"

class fastreid : public Feature {
public:
    explicit fastreid(const YAML::Node &config);
};

#endif //TENSORRT_INFERENCE_FAST_REID_H
