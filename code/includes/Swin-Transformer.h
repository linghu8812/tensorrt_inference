#ifndef TENSORRT_INFERENCE_SWIN_TRANSFORMER_H
#define TENSORRT_INFERENCE_SWIN_TRANSFORMER_H

#include "classification.h"

class Swin_Transformer : public Classification {
public:
    explicit Swin_Transformer(const YAML::Node &config);
};

#endif //TENSORRT_INFERENCE_SWIN_TRANSFORMER_H
