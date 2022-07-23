#ifndef TENSORRT_INFERENCE_FACES_H
#define TENSORRT_INFERENCE_FACES_H

#include "detection.h"
#include "keypoints.h"

struct FaceBox{
    Bbox bbox{};
    KeyPointsRes key_points;
    bool has_mask = false;
};

struct FacesRes{
    std::vector<FaceBox> faces_results;
};

class Faces : public Model
{
public:
    explicit Faces(const YAML::Node &config);
    std::vector<FacesRes> InferenceImages(std::vector<cv::Mat> &vec_img);
    void InferenceFolder(const std::string &folder_name) override;
    void DrawResults(const std::vector<FacesRes> &results, std::vector<cv::Mat> &vec_img,
                     std::vector<std::string> image_names);

protected:
    void NmsDetect(std::vector<FaceBox> &detections);
    virtual std::vector<FacesRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output)=0;
    virtual void GenerateAnchors()=0;
    float obj_threshold;
    float nms_threshold;
    bool detect_mask;
    float mask_thresh;
    float landmark_std;

    int bbox_head = 3;
    int landmark_head = 10;
    std::vector<int> feature_sizes;
    std::vector<int> feature_steps;
    std::vector<std::vector<int>> feature_maps;
    int sum_of_feature;
};

#endif //TENSORRT_INFERENCE_FACES_H
