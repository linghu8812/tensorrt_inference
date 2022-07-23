//
// Created by linghu8812 on 2022/7/21.
//

#include "YOLO.h"

YOLO::YOLO(const YAML::Node &config) : Detection(config) {}

std::vector<DetectRes> YOLO::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<DetectRes> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat)
    {
        DetectRes result;
        float ratio = float(src_img.cols) / float(IMAGE_WIDTH) > float(src_img.rows) / float(IMAGE_HEIGHT)  ? float(src_img.cols) / float(IMAGE_WIDTH) : float(src_img.rows) / float(IMAGE_HEIGHT);
        float *out = output + index * outSize;
        for (int position = 0; position < num_rows; position++) {
            float *row = out + position * (CATEGORY + 5);
            Bbox box;
            if (row[4] < obj_threshold)
                continue;
            auto max_pos = std::max_element(row + 5, row + CATEGORY + 5);
            box.prob = row[4] * row[max_pos - row];
            box.classes = max_pos - row - 5;
            box.x = row[0] * ratio;
            box.y = row[1] * ratio;
            box.w = row[2] * ratio;
            box.h = row[3] * ratio;
            result.det_results.push_back(box);
        }
        NmsDetect(result.det_results);
        vec_result.push_back(result);
        index++;
    }
    return vec_result;
}