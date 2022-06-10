#ifndef SCRFD_TRT_SCRFD_H
#define SCRFD_TRT_SCRFD_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class scrfd{
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
        bool has_mask = false;
    };

public:
    explicit scrfd(const std::string &config_file);
    ~scrfd();
    void LoadEngine();
    bool InferenceFolder(const std::string &folder_name);
private:
    void EngineInference(const std::vector<std::string> &image_list, const int &outSize,void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream);
    void GenerateAnchors();
    std::vector<float> prepareImage(std::vector<cv::Mat> & vec_img);
    std::vector<std::vector<FaceRes>> postProcess(const std::vector<cv::Mat> &vec_Mat, float *output, const int &outSize);
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
    float mean;
    float std;
    float obj_threshold;
    float nms_threshold;
    bool detect_mask;
    float mask_thresh;
    float landmark_std;

    cv::Mat refer_matrix;
    int anchor_num = 2;
    int bbox_head = 3;
    int landmark_head = 10;
    std::vector<int> feature_sizes;
    std::vector<int> feature_steps;
    std::vector<std::vector<int>> feature_maps;
    std::vector<std::vector<int>> anchor_sizes;
    int sum_of_feature;
};

#endif //SCRFD_TRT_SCRFD_H
