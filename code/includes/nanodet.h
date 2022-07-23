#ifndef TENSORRT_INFERENCE_NANODET_H
#define TENSORRT_INFERENCE_NANODET_H

#include "detection.h"

class nanodet : public Detection {
public:
    explicit nanodet(const YAML::Node &config);

private:
    std::vector<DetectRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) override;
    int refer_rows;
    int refer_cols;
    cv::Mat refer_matrix;
};

#endif //TENSORRT_INFERENCE_NANODET_H
