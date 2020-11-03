#ifndef FACE_ALIGNMENT_TRT_FACE_ALIGNMENT_H
#define FACE_ALIGNMENT_TRT_FACE_ALIGNMENT_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class FaceAlignment
{
public:
    explicit FaceAlignment(const std::string &config_file);
    ~FaceAlignment();
    void LoadEngine();
    bool InferenceFolder(const std::string &folder_name);

private:
    void EngineInference(const std::vector<std::string> &image_list, const int &outSize,void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream);
    std::vector<float> prepareImage(std::vector<cv::Mat> & vec_img);
    std::vector<std::vector<cv::Point2f>> postProcess(const std::vector<cv::Mat> &vec_Mat, float *output, const int &outSize);
    std::string onnx_file;
    std::string engine_file;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
};

#endif //FACE_ALIGNMENT_TRT_FACE_ALIGNMENT_H
