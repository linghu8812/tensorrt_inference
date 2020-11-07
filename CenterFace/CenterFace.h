#ifndef CENTERFACE_CENTERFACE_H
#define CENTERFACE_CENTERFACE_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class CenterFace{
    struct FaceBox{
        float x;
        float y;
        float w;
        float h;
    };

    struct FaceRes{
        float confidence;
        FaceBox face_box;
        std::vector<cv::Point2f> keypoints;
    };

public:
    explicit CenterFace(const std::string &config_file);
    ~CenterFace();
    void LoadEngine();
    bool InferenceFolder(const std::string &folder_name);
private:
    void EngineInference(const std::vector<std::string> &image_list, const int *outSize,void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream);
    std::vector<float> prepareImage(std::vector<cv::Mat> & vec_img);
    std::vector<std::vector<FaceRes>> postProcess(const std::vector<cv::Mat> &vec_Mat,
            float *output_1, float *output_2, float *output_3, float *output_4,
            const int &outSize_1, const int &outSize_2, const int &outSize_3, const int &outSize_4);
    void NmsDetect(std::vector<FaceRes> &detections);
    static float IOUCalculate(const FaceBox &det_a, const FaceBox &det_b);

    std::string onnx_file;
    std::string engine_file;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    float obj_threshold;
    float nms_threshold;
};

#endif //CENTERFACE_CENTERFACE_H
