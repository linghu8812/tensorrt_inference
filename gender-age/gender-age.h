#ifndef GENDER_AGE_TRT_GENDER_AGE_H
#define GENDER_AGE_TRT_GENDER_AGE_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class GenderAge {

    struct attribute
    {
        int gender;
        int age;
    };

public:
    GenderAge(const std::string &config_file);
    ~GenderAge();
    void LoadEngine();
    bool InferenceFolder(const std::string &folder_name);

private:
    void EngineInference(const std::vector<std::string> &image_list, const int &outSize, void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream);
    std::vector<float> prepareImage(std::vector<cv::Mat> &image);
    std::vector<attribute> postProcess(float out[], const int &MAT_SIZE, const int &outSize);

    std::string onnx_file;
    std::string engine_file;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
};

#endif //GENDER_AGE_TRT_GENDER_AGE_H
