#ifndef TENSORRT_INFERENCE_GENDER_AGE_H
#define TENSORRT_INFERENCE_GENDER_AGE_H

#include "model.h"

struct attribute{
    int gender;
    int age;
};

class GenderAge : public Model {
public:
    explicit GenderAge(const YAML::Node &config);
    std::vector<attribute> InferenceImages(std::vector<cv::Mat> &vec_img);
    void InferenceFolder(const std::string &folder_name) override;
    void DrawResults(const std::vector<attribute> &results, std::vector<cv::Mat> &vec_img,
                     std::vector<std::string> image_names);

private:
    std::vector<attribute> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output);
};

#endif //TENSORRT_INFERENCE_GENDER_AGE_H
