#ifndef MMPOSE_TRT_MMPOSE_H
#define MMPOSE_TRT_MMPOSE_H

#include "model.h"

class mmpose : public Model
{
    struct KeyPoint{
        int x;
        int y;
        int number;
        float prob;
    };

public:
    mmpose(const std::string &config_file);
    ~mmpose();
    bool InferenceFolder(const std::string &folder_name) override;

private:
    void EngineInference(const std::vector<std::string> &image_list, const int &outSize,void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream) override;
    std::vector<float> prepareImage(std::vector<cv::Mat> & vec_img) override;
    std::vector<std::vector<KeyPoint>> postProcess(const std::vector<cv::Mat> &vec_Mat, float *output, const int &outSize);

    int num_key_points;
    std::vector<std::vector<int>> skeleton;
    float point_thresh;
};

#endif //MMPOSE_TRT_MMPOSE_H
