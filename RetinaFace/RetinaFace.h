#ifndef RETINAFACE_TRT_RETINAFACE_H
#define RETINAFACE_TRT_RETINAFACE_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class RetinaFace{
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
    RetinaFace(const std::string &config_file);
    ~RetinaFace();
    void LoadEngine();
    bool InferenceFolder(const std::string &folder_name);
private:
    void EngineInference(const std::vector<std::string> &image_list, const int *outSize,void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream);
    void GenerateAnchors();
    std::vector<float> prepareImage(std::vector<cv::Mat> & vec_img);
    std::vector<std::vector<FaceRes>> postProcess(const std::vector<cv::Mat> &vec_Mat, float *output_1, float *output_2, float *output_3,
            const int &outSize_1, const int &outSize_2, const int &outSize_3);
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
    cv::Mat refer_matrix;

    int anchor_num = 2;
    int bbox_head = 4;
    int landmark_head = 10;
    int class_head = 2;
    int out1_step;
    int out2_step;
    int out3_step;
    std::vector<int> feature_sizes;
    std::vector<int> feature_steps;
    std::vector<int> feature_maps;
    std::vector<std::vector<int>> anchor_sizes;
};

#endif //RETINAFACE_TRT_RETINAFACE_H
