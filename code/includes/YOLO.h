//
// Created by linghu8812 on 2022/7/21.
//

#ifndef TENSORRT_INFERENCE_YOLO_H
#define TENSORRT_INFERENCE_YOLO_H

#include "detection.h"

class YOLO : public Detection {
public:
    explicit YOLO(const YAML::Node &config);

protected:
    std::vector<DetectRes> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) override;
};
#endif //TENSORRT_INFERENCE_YOLO_H
