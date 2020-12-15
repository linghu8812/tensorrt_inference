#ifndef MINIFASNET_TRT_MINIFASNET_H
#define MINIFASNET_TRT_MINIFASNET_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class MiniFASNet
{
public:
    MiniFASNet(const std::string &config_file);
    ~MiniFASNet();
    void LoadEngine();
    bool InferenceFolder(const std::string &folder_name);

private:
    void EngineInference(const std::vector<std::string> &image_list, const int &outSize,void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream);
    std::vector<float> prepareImage(std::vector<cv::Mat> & vec_img);
    std::string onnx_file;
    std::string engine_file;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
};

#endif //MINIFASNET_TRT_MINIFASNET_H
