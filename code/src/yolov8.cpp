#include "yolov8.h"

YOLOv8::YOLOv8(const YAML::Node &config) : Detection(config) {}

std::vector<DetectRes> YOLOv8::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<DetectRes> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat) {
        DetectRes result;
        float ratio = float(src_img.cols) / float(IMAGE_WIDTH) > float(src_img.rows) / float(IMAGE_HEIGHT)  ? float(src_img.cols) / float(IMAGE_WIDTH) : float(src_img.rows) / float(IMAGE_HEIGHT);
        float *out = output + index * outSize;
        cv::Mat res_mat = cv::Mat(CATEGORY + 4, num_rows, CV_32FC1, out);
//        std::cout << res_mat << std::endl;
        res_mat = res_mat.t();
//        out = res_mat.ptr<float>(0);
        cv::Mat prob_mat;
        cv::reduce(res_mat.colRange(4, CATEGORY + 4), prob_mat, 1, cv::REDUCE_MAX);
        out = res_mat.ptr<float>(0);
        for (int position = 0; position < num_rows; position++) {
            float *row = out + position * (CATEGORY + 4);
            Bbox box;
            box.prob = *prob_mat.ptr<float>(position);
            if (box.prob < obj_threshold)
                continue;
            auto max_pos = std::max_element(row + 4, row + CATEGORY + 4);
            box.classes = max_pos - row - 4;
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