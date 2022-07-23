#ifndef TENSORRT_INFERENCE_DETECTION_H
#define TENSORRT_INFERENCE_DETECTION_H

#include "classification.h"

struct Bbox : ClassRes{
    float x;
    float y;
    float w;
    float h;
};

struct DetectRes {
    std::vector<Bbox> det_results;
};

class Detection : public Model
{
public:
    explicit Detection(const YAML::Node &config);
    std::vector<DetectRes> InferenceImages(std::vector<cv::Mat> &vec_img);
    void InferenceFolder(const std::string &folder_name) override;
    void DrawResults(const std::vector<DetectRes> &detections, std::vector<cv::Mat> &vec_img,
                     std::vector<std::string> image_name);
    static float IOUCalculate(const Bbox &det_a, const Bbox &det_b);

protected:
    virtual std::vector<DetectRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output)=0;
    void NmsDetect(std::vector<Bbox> &detections);
    std::map<int, std::string> class_labels;
    int CATEGORY;
    float obj_threshold;
    float nms_threshold;
    bool agnostic;
    std::vector<cv::Scalar> class_colors;
    std::vector<int> strides;
    std::vector<int> num_anchors;
    int num_rows = 0;
};

#endif //TENSORRT_INFERENCE_DETECTION_H
