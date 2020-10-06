#ifndef ALEXNET_TRT_ALEXNET_H
#define ALEXNET_TRT_ALEXNET_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class SEResNeXt
{
public:
    SEResNeXt(const std::string &config_file);
    ~SEResNeXt();
    void LoadEngine();
    bool InferenceFolder(const std::string &folder_name);

private:
    void EngineInference(const std::vector<std::string> &image_list, const int &outSize,void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream);
    std::vector<float> prepareImage(std::vector<cv::Mat> & vec_img);
    std::string onnx_file;
    std::string engine_file;
    std::string labels_file;
    std::map<int, std::string> imagenet_labels;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    std::vector<float> img_mean;
    std::vector<float> img_std;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
};

#endif //ALEXNET_TRT_ALEXNET_H
