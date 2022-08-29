//
// Created by linghu8812 on 2022/8/29.
//

#include "build.h"

std::shared_ptr<Model> build_model(char **argv) {
    std::string model_arch = argv[1];
    std::string config_file = argv[2];
    YAML::Node root = YAML::LoadFile(config_file);
    auto model = std::shared_ptr<Model>();
    if (model_arch == "yolov5")
        model = std::make_shared<YOLOv5>(root[model_arch]);
    else if (model_arch == "yolov5_cls")
        model = std::make_shared<YOLOv5_cls>(root[model_arch]);
    return model;
}

