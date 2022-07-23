//
// Created by linghu8812 on 2022/7/23.
//

#ifndef TENSORRT_INFERENCE_SEGMENTATION_H
#define TENSORRT_INFERENCE_SEGMENTATION_H

#include "model.h"

struct SegmentationRes {
    cv::Mat seg_result;
};

class Segmentation : public Model
{
public:
    explicit Segmentation(const YAML::Node &config);
    std::vector<SegmentationRes> InferenceImages(std::vector<cv::Mat> &vec_img);
    void InferenceFolder(const std::string &folder_name) override;
    void DrawResults(const std::vector<SegmentationRes> &detections, std::vector<cv::Mat> &vec_img,
                     std::vector<std::string> image_name);

protected:
    virtual std::vector<SegmentationRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output)=0;
    int OUT_WIDTH;
    int OUT_HEIGHT;
    int CATEGORY;
    std::vector<cv::Scalar> class_colors;
};

#endif //TENSORRT_INFERENCE_SEGMENTATION_H
