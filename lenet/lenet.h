#ifndef LENET_TRT_LENET_H
#define LENET_TRT_LENET_H

#include "model.h"

class LeNet : public Model
{
public:
    LeNet(const std::string &config_file);
    ~LeNet();
    bool InferenceFolder(const std::string &folder_name) override;

private:
    void EngineInference(const std::vector<std::string> &image_list, const int &outSize,void **buffers,
            const std::vector<int64_t> &bufferSize, cudaStream_t stream) override;
    std::vector<float> prepareImage(std::vector<cv::Mat> & image) override;
};

#endif //LENET_TRT_LENET_H
