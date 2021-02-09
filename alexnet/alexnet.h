#ifndef ALEXNET_TRT_ALEXNET_H
#define ALEXNET_TRT_ALEXNET_H

#include "model.h"

class AlexNet : public Model
{
public:
    AlexNet(const std::string &config_file);
    ~AlexNet();
    bool InferenceFolder(const std::string &folder_name) override;

private:
    void EngineInference(const std::vector<std::string> &image_list, const int &outSize,void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream) override;
    std::vector<float> prepareImage(std::vector<cv::Mat> & vec_img) override;
    std::map<int, std::string> imagenet_labels;
};

#endif //ALEXNET_TRT_ALEXNET_H
