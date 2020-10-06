#ifndef ARCFACE_TRT_ARCFACE_H
#define ARCFACE_TRT_ARCFACE_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class ArcFace {
public:
    ArcFace(const std::string &config_file);
    ~ArcFace();
    void LoadEngine();
    bool InferenceFolder(const std::string &folder_name);

private:
    void EngineInference(const std::vector<std::string> &image_list, const int &outSize, void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream);
    std::vector<float> prepareImage(std::vector<cv::Mat> &image);
    void ReshapeandNormalize(float out[], cv::Mat &feature, const int &MAT_SIZE, const int &outSize);

    std::string onnx_file;
    std::string engine_file;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
};

#endif //ARCFACE_TRT_ARCFACE_H
