//
// Created by linghu8812 on 2022/7/19.
//

#ifndef TENSORRT_INFERENCE_KEYPOINTS_H
#define TENSORRT_INFERENCE_KEYPOINTS_H

#include "model.h"

struct KeyPoint{
    int x;
    int y;
};

struct KeyPointsRes{
    std::vector<KeyPoint> key_points;
};

class KeyPoints : public Model
{
public:
    explicit KeyPoints(const YAML::Node &config);
    std::vector<KeyPointsRes> InferenceImages(std::vector<cv::Mat> &vec_img);
    void InferenceFolder(const std::string &folder_name) override;
    void DrawResults(const std::vector<KeyPointsRes> &results, std::vector<cv::Mat> &vec_img,
                     std::vector<std::string> image_names);

protected:
    std::vector<KeyPointsRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output);
};

#endif //TENSORRT_INFERENCE_KEYPOINTS_H
