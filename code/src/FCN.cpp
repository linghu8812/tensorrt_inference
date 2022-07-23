#include "FCN.h"

FCN::FCN(const YAML::Node &config) : Segmentation(config) {}

std::vector<SegmentationRes> FCN::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<SegmentationRes> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat) {
        SegmentationRes result;
        result.seg_result = cv::Mat(OUT_WIDTH, OUT_HEIGHT, CV_8UC1);
        float *out = output + index * outSize;
        for (int h = 0; h < OUT_HEIGHT; h++) {
            uchar *row = result.seg_result.ptr<uchar>(h);
            for (int w = 0; w < OUT_WIDTH; w++) {
                float *current = out + (h * OUT_WIDTH + w) * (CATEGORY + 1);
                auto max_pos = std::max_element(current, current + CATEGORY + 1);
                row[w] = max_pos - current;
            }
        }
        vec_result.push_back(result);
        index++;
    }
    return vec_result;
}
