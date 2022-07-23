#ifndef TENSORRT_INFERNECE_MMPOSE_H
#define TENSORRT_INFERNECE_MMPOSE_H

#include "keypoints.h"

struct mmposePoint : KeyPoint{
    int number;
    float prob;
};

struct mmposeRes{
    std::vector<mmposePoint> key_points;
};

class mmpose : public KeyPoints {
public:
    explicit mmpose(const YAML::Node &config);
    std::vector<mmposeRes> InferenceImages(std::vector<cv::Mat> &vec_img);
    void InferenceFolder(const std::string &folder_name) override;
    void DrawResults(const std::vector<mmposeRes> &results, std::vector<cv::Mat> &vec_img,
                     std::vector<std::string> image_names);

private:
    std::vector<mmposeRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output);
    int num_key_points;
    std::vector<std::vector<int>> skeleton;
    float point_thresh;
};

#endif //TENSORRT_INFERNECE_MMPOSE_H
