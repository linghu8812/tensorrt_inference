#ifndef YOLOV4_TRT_YOLOV4_H
#define YOLOV4_TRT_YOLOV4_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class ScaledYOLOv4
{
    struct DetectRes{
        int classes;
        float x;
        float y;
        float w;
        float h;
        float prob;
    };

public:
    explicit ScaledYOLOv4(const std::string &config_file);
    ~ScaledYOLOv4();
    void LoadEngine();
    bool InferenceFolder(const std::string &folder_name);

private:
    void EngineInference(const std::vector<std::string> &image_list, const int &outSize,void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream);
    void GenerateReferMatrix();
    std::vector<float> prepareImage(std::vector<cv::Mat> & vec_img);
    std::vector<std::vector<DetectRes>> postProcess(const std::vector<cv::Mat> &vec_Mat, float *output, const int &outSize);
    void NmsDetect(std::vector <DetectRes> &detections);
    float IOUCalculate(const DetectRes &det_a, const DetectRes &det_b);

    std::string onnx_file;
    std::string engine_file;
    std::string labels_file;
    std::map<int, std::string> detect_labels;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    int CATEGORY;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    float obj_threshold;
    float nms_threshold;
    int refer_rows;
    int refer_cols;
    cv::Mat refer_matrix;
    std::vector<int> strides;
    std::vector<int> num_anchors;
    std::vector<std::vector<int>> anchors;
    std::vector<std::vector<int>> grids;
    std::vector<cv::Scalar> class_colors;
};

#endif //YOLOV4_TRT_YOLOV4_H
