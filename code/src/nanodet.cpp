#include "nanodet.h"

nanodet::nanodet(const YAML::Node &config) : Detection(config) {
    refer_rows = 0;
    refer_cols = 3;
    for (const int &stride : strides) {
        refer_rows += IMAGE_WIDTH * IMAGE_HEIGHT / stride / stride;
    }
    int index = 0;
    refer_matrix = cv::Mat(refer_rows, refer_cols, CV_32FC1);
    for (const int &stride : strides) {
        for (int h = 0; h < IMAGE_HEIGHT / stride; h++)
            for (int w = 0; w < IMAGE_WIDTH / stride; w++) {
                auto *row = refer_matrix.ptr<float>(index);
                row[0] = float((2 * w + 1) * stride - 1) / 2;
                row[1] = float((2 * h + 1) * stride - 1) / 2;
                row[2] = stride;
                index += 1;
            }
    }
}

std::vector<DetectRes> nanodet::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<DetectRes> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat) {
        DetectRes result;
        float *out = output + index * outSize;
        float ratio = std::max(float(src_img.cols) / float(IMAGE_WIDTH), float(src_img.rows) / float(IMAGE_HEIGHT));
        cv::Mat result_matrix = cv::Mat(refer_rows, CATEGORY + 4, CV_32FC1, out);
        for (int row_num = 0; row_num < refer_rows; row_num++) {
            Bbox box{};
            auto *row = result_matrix.ptr<float>(row_num);
            auto max_pos = std::max_element(row + 4, row + CATEGORY + 4);
            box.prob = row[max_pos - row];
            if (box.prob < obj_threshold)
                continue;
            box.classes = max_pos - row - 4;
            auto *anchor = refer_matrix.ptr<float>(row_num);
            box.x = (anchor[0] - row[0] * anchor[2] + anchor[0] + row[2] * anchor[2]) / 2 * ratio;
            box.y = (anchor[1] - row[1] * anchor[2] + anchor[1] + row[3] * anchor[2]) / 2 * ratio;
            box.w = (row[2] + row[0]) * anchor[2] * ratio;
            box.h = (row[3] + row[1]) * anchor[2] * ratio;
            result.det_results.push_back(box);
        }
        NmsDetect(result.det_results);
        vec_result.push_back(result);
        index++;
    }
    return vec_result;
}
