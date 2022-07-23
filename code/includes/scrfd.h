#ifndef TENSORRT_INFERENCE_SCRFD_H
#define TENSORRT_INFERENCE_SCRFD_H

#include "faces.h"

class scrfd : public Faces{
public:
    explicit scrfd(const YAML::Node &config);

private:
    std::vector<FacesRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) override;
    void GenerateAnchors() override;

    int anchor_num = 2;
    cv::Mat refer_matrix;
    std::vector<std::vector<int>> anchor_sizes;
};

#endif //TENSORRT_INFERENCE_SCRFD_H
