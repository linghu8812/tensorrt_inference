#ifndef YOLOV7_TRT_YOLOV7_H
#define YOLOV7_TRT_YOLOV7_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"

#define EXPORT_COCO_JSON 0
#define WRITE_IMG 1
#define VERBOSE 1

#if EXPORT_COCO_JSON
//https://github.com/ultralytics/yolov5/blob/d833ab3d2529626d4cc4c6ae28ce7858b9ca738f/utils/general.py#L321

static inline std::vector<size_t> getCoco80ToCoco91Class() {
    return {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};
}

#include "json.hpp"
using json = nlohmann::json;
#define WRITE_IMG 0
#endif

class yolov7 {

    struct DetectRes {
        int classes;
        float x;
        float y;
        float w;
        float h;
        float prob;
    };

public:
    yolov7(const std::string &config_file);
    ~yolov7();
    void LoadEngine();
    bool InferenceFolder(const std::string &folder_name);

private:
    void EngineInference(const std::vector<std::string> &image_list, const int &outSize, void **buffers,
            const std::vector<int64_t> &bufferSize, cudaStream_t stream);
    std::vector<float> prepareImage(std::vector<cv::Mat> & vec_img);
    std::vector<std::vector<DetectRes>> postProcess(const std::vector<cv::Mat> &vec_Mat, float *output, const int &outSize);
    void NmsDetect(std::vector <DetectRes> &detections);
    float IOUCalculate(const DetectRes &det_a, const DetectRes &det_b);
    std::string onnx_file;
    std::string engine_file;
    std::string labels_file;
    std::map<int, std::string> coco_labels;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    int CATEGORY;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    float obj_threshold;
    float nms_threshold;
    std::vector<int> strides;
    std::vector<int> num_anchors;
    int num_rows = 0;
    std::vector<cv::Scalar> class_colors;
};

#endif //YOLOV7_TRT_YOLOV7_H
