#ifndef TENSORRT_INFERENCE_CLASSIFICATION_H
#define TENSORRT_INFERENCE_CLASSIFICATION_H

#include "model.h"

struct ClassRes{
    int classes;
    float prob;
};

class Classification : public Model
{
public:
    explicit Classification(const YAML::Node &config);
    std::vector<ClassRes> InferenceImages(std::vector<cv::Mat> &vec_img);
    void InferenceFolder(const std::string &folder_name) override;
    void DrawResults(const std::vector<ClassRes> &results, std::vector<cv::Mat> &vec_img,
                     std::vector<std::string> image_names);

protected:
    std::vector<ClassRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output);
    std::map<int, std::string> class_labels;
    int CATEGORY;
};

#endif //TENSORRT_INFERENCE_CLASSIFICATION_H
