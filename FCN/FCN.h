#ifndef FCN_TRT_FCN_H
#define FCN_TRT_FCN_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class FCN
{
public:
    FCN(const std::string &config_file);
    ~FCN();
    void LoadEngine();
    bool InferenceFolder(const std::string &folder_name);

private:
    void EngineInference(const std::vector<std::string> &image_list, const int &outSize,void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream);
    std::vector<float> prepareImage(std::vector<cv::Mat> & vec_img);
    std::vector<cv::Mat> postProcess(const std::vector<cv::Mat> &vec_Mat, float *output, const int &outSize);
    std::string onnx_file;
    std::string engine_file;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    int OUT_WIDTH;
    int OUT_HEIGHT;
    int CATEGORY;
    std::vector<float> img_mean;
    std::vector<float> img_std;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    std::vector<cv::Scalar> class_colors;
};

#endif //FCN_TRT_FCN_H
