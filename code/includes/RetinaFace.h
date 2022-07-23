#ifndef TENSORRT_INFERENCE_RETINAFACE_H
#define TENSORRT_INFERENCE_RETINAFACE_H

#include "faces.h"

class RetinaFace : public Faces{
public:
    explicit RetinaFace(const YAML::Node &config);

private:
    std::vector<FacesRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) override;
    void GenerateAnchors() override;

    int anchor_num = 2;
    cv::Mat refer_matrix;
    std::vector<std::vector<int>> anchor_sizes;
};

#endif //TENSORRT_INFERENCE_RETINAFACE_H
