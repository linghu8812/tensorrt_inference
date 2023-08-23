//
// Created by linghu8812 on 2022/8/29.
//

#include "build.h"

std::shared_ptr<Model> build_model(char **argv) {
    std::string model_arch = argv[1];
    std::string config_file = argv[2];
    YAML::Node root = YAML::LoadFile(config_file);
    auto model = std::shared_ptr<Model>();
    if (model_arch == "alexnet")
        model = std::make_shared<AlexNet>(root[model_arch]);
    else if (model_arch == "arcface")
        model = std::make_shared<ArcFace>(root[model_arch]);
    else if (model_arch == "CenterFace")
        model = std::make_shared<CenterFace>(root[model_arch]);
    else if (model_arch == "efficientnet")
        model = std::make_shared<EfficientNet>(root[model_arch]);
    else if (model_arch == "face_alignment")
        model = std::make_shared<FaceAlignment>(root[model_arch]);
    else if (model_arch == "fastreid")
        model = std::make_shared<fastreid>(root[model_arch]);
    else if (model_arch == "FCN")
        model = std::make_shared<FCN>(root[model_arch]);
    else if (model_arch == "gender-age")
        model = std::make_shared<GenderAge>(root[model_arch]);
    else if (model_arch == "ghostnet")
        model = std::make_shared<GhostNet>(root[model_arch]);
    else if (model_arch == "lenet")
        model = std::make_shared<LeNet>(root[model_arch]);
    else if (model_arch == "MiniFASNet")
        model = std::make_shared<MiniFASNet>(root[model_arch]);
    else if (model_arch == "mmpose")
        model = std::make_shared<mmpose>(root[model_arch]);
    else if (model_arch == "nanodet")
        model = std::make_shared<nanodet>(root[model_arch]);
    else if (model_arch == "RetinaFace")
        model = std::make_shared<RetinaFace>(root[model_arch]);
    else if (model_arch == "ScaledYOLOv4")
        model = std::make_shared<ScaledYOLOv4>(root[model_arch]);
    else if (model_arch == "scrfd")
        model = std::make_shared<scrfd>(root[model_arch]);
    else if (model_arch == "seresnext")
        model = std::make_shared<SEResNeXt>(root[model_arch]);
    else if (model_arch == "Swin_Transformer")
        model = std::make_shared<Swin_Transformer>(root[model_arch]);
    else if (model_arch == "yolor")
        model = std::make_shared<yolor>(root[model_arch]);
    else if (model_arch == "Yolov4")
        model = std::make_shared<YOLOv4>(root[model_arch]);
    else if (model_arch == "yolov5")
        model = std::make_shared<YOLOv5>(root[model_arch]);
    else if (model_arch == "yolov5_cls")
        model = std::make_shared<YOLOv5_cls>(root[model_arch]);
    else if (model_arch == "YOLOv6")
        model = std::make_shared<YOLOv6>(root[model_arch]);
    else if (model_arch == "yolov7")
        model = std::make_shared<yolov7>(root[model_arch]);
    else if (model_arch == "yolov8")
        model = std::make_shared<YOLOv8>(root[model_arch]);
    else
        std::cout << "No model arch matched!" << std::endl;
    return model;
}

