//
// Created by linghu8812 on 2022/8/29.
//

#ifndef TENSORRT_INFERENCE_BUILD_H
#define TENSORRT_INFERENCE_BUILD_H

#include "yolov5.h"

std::shared_ptr<Model> build_model(char **argv);

#endif //TENSORRT_INFERENCE_BUILD_H
