#ifndef TENSORRT_INFERENCE_CENTERFACE_H
#define TENSORRT_INFERENCE_CENTERFACE_H

#include "faces.h"

class CenterFace : public Faces{
public:
    explicit CenterFace(const YAML::Node &config);

private:
    void GenerateAnchors() override;
    std::vector<FacesRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) override;
};

#endif //TENSORRT_INFERENCE_CENTERFACE_H
