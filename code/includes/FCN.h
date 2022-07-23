#ifndef TENSORRT_INFERENCE_FCN_H
#define TENSORRT_INFERENCE_FCN_H

#include "segmentation.h"

class FCN : public Segmentation {
public:
    explicit FCN(const YAML::Node &config);

private:
    std::vector<SegmentationRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output);
};

#endif //TENSORRT_INFERENCE_FCN_H
