#ifndef NANODET_NANODET_H
#define NANODET_NANODET_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class nanodet{
public:
    struct DetectRes{
        int classes;
        float x;
        float y;
        float w;
        float h;
        float prob;
    };

public:
    explicit nanodet(const std::string &config_file);
    ~nanodet();
    void LoadEngine();
    bool InferenceFolder(const std::string &folder_name);
private:
    void EngineInference(const std::vector<std::string> &image_list, const int &outSize,void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream);
    void GenerateReferMatrix();
    std::vector<float> prepareImage(std::vector<cv::Mat> & vec_img);
    std::vector<std::vector<DetectRes>> postProcess(const std::vector<cv::Mat> &vec_Mat, float *output, const int &outSize);
    void NmsDetect(std::vector<DetectRes> &detections);
    static float IOUCalculate(const DetectRes &det_a, const DetectRes &det_b);

    std::string onnx_file;
    std::string engine_file;
    std::string labels_file;
    std::map<int, std::string> detect_labels;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    int CATEGORY;
    std::vector<float> img_mean;
    std::vector<float> img_std;
    float obj_threshold;
    float nms_threshold;
    std::vector<int> strides;
    std::vector<cv::Scalar> class_colors;
    int refer_rows;
    int refer_cols;
    cv::Mat refer_matrix;
};

#endif //NANODET_NANODET_H
