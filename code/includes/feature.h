//
// Created by linghu8812 on 2022/7/18.
//

#ifndef TENSORRT_INFERENCE_FEATURE_H
#define TENSORRT_INFERENCE_FEATURE_H

#include "model.h"

struct FeatureRes{
    std::vector<float> feature;
};

class Feature : public Model
{
public:
    explicit Feature(const YAML::Node &config);
    std::vector<FeatureRes> InferenceImages(std::vector<cv::Mat> &vec_img);
    void InferenceFolder(const std::string &folder_name) override;
    void ComputeSimilarity(const std::vector<FeatureRes> &results_a, const std::vector<FeatureRes> &results_b);

protected:
    std::vector<FeatureRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output);
    cv::Mat Feature2Mat(const std::vector<FeatureRes> &vec_results);
};

#endif //TENSORRT_INFERENCE_FEATURE_H
