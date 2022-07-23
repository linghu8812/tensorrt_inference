#ifndef TENSORRT_INFERNECE_FACE_ALIGNMENT_H
#define TENSORRT_INFERNECE_FACE_ALIGNMENT_H

#include "keypoints.h"

class FaceAlignment : public KeyPoints {
public:
    explicit FaceAlignment(const YAML::Node &config);
};

#endif //TENSORRT_INFERNECE_FACE_ALIGNMENT_H
