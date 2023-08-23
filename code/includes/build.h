//
// Created by linghu8812 on 2022/8/29.
//

#ifndef TENSORRT_INFERENCE_BUILD_H
#define TENSORRT_INFERENCE_BUILD_H

#include "alexnet.h"
#include "arcface.h"
#include "CenterFace.h"
#include "efficientnet.h"
#include "face_alignment.h"
#include "fast-reid.h"
#include "FCN.h"
#include "gender-age.h"
#include "ghostnet.h"
#include "lenet.h"
#include "MiniFASNet.h"
#include "mmpose.h"
#include "nanodet.h"
#include "RetinaFace.h"
#include "ScaledYOLOv4.h"
#include "scrfd.h"
#include "seresnext.h"
#include "Swin-Transformer.h"
#include "yolor.h"
#include "Yolov4.h"
#include "yolov5.h"
#include "YOLOv6.h"
#include "yolov7.h"
#include "yolov8.h"

std::shared_ptr<Model> build_model(char **argv);

#endif //TENSORRT_INFERENCE_BUILD_H
