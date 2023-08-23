#ifndef TENSORRT_INFERENCE_YOLOV8_H
#define TENSORRT_INFERENCE_YOLOV8_H

#include "detection.h"

class YOLOv8 : public Detection {
public:
    explicit YOLOv8(const YAML::Node &config);

private:
    std::vector<DetectRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) override;
};

#endif //TENSORRT_INFERENCE_YOLOV8_H
