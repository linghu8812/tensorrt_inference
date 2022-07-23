#include "Yolov4.h"

YOLOv4::YOLOv4(const YAML::Node &config) : Detection(config) {
    anchors = config["anchors"].as<std::vector<std::vector<int>>>();
    int index = 0;
    for (const int &stride : strides)
    {
        grids.push_back({num_anchors[index], int(IMAGE_HEIGHT / stride), int(IMAGE_WIDTH / stride)});
    }
    refer_rows = 0;
    refer_cols = 6;
    for (const std::vector<int> &grid : grids) {
        refer_rows += std::accumulate(grid.begin(), grid.end(), 1, std::multiplies<int>());
    }
    GenerateReferMatrix();
    letter_box = resize == "keep_ratio";
    class_colors.resize(CATEGORY);
    srand((int) time(nullptr));
    for (cv::Scalar &class_color : class_colors)
        class_color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
}

void YOLOv4::GenerateReferMatrix() {
    refer_matrix = cv::Mat(refer_rows, refer_cols, CV_32FC1);
    int position = 0;
    for (int n = 0; n < (int)grids.size(); n++) {
        for (int c = 0; c < grids[n][0]; c++) {
            std::vector<int> anchor = anchors[n * grids[n][0] + c];
            for (int h = 0; h < grids[n][1]; h++)
                for (int w = 0; w < grids[n][2]; w++) {
                    auto *row = refer_matrix.ptr<float>(position);
                    row[0] = w;
                    row[1] = grids[n][2];
                    row[2] = h;
                    row[3] = grids[n][1];
                    row[4] = anchor[0];
                    row[5] = anchor[1];
                    position++;
                }
        }
    }
}

std::vector<DetectRes> YOLOv4::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<DetectRes> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat) {
        DetectRes result;
        float *out = output + index * outSize;
        float ratio = std::max(float(src_img.cols) / float(IMAGE_WIDTH), float(src_img.rows) / float(IMAGE_HEIGHT));
        cv::Mat result_matrix = cv::Mat(refer_rows, CATEGORY + 5, CV_32FC1, out);
        for (int row_num = 0; row_num < refer_rows; row_num++) {
            Bbox box{};
            auto *row = result_matrix.ptr<float>(row_num);
            if ((letter_box ? row[4] : sigmoid(row[4])) < obj_threshold)
                continue;
            auto max_pos = std::max_element(row + 5, row + CATEGORY + 5);
            box.prob = letter_box ? row[4] * row[max_pos - row] : sigmoid(row[4]) * sigmoid(row[max_pos - row]);
            box.classes = max_pos - row - 5;
            auto *anchor = refer_matrix.ptr<float>(row_num);
            box.x = letter_box ? (float)(row[0] * 2 - 0.5 + anchor[0]) / anchor[1] * (float)IMAGE_WIDTH * ratio : (sigmoid(row[0]) + anchor[0]) / anchor[1] * (float)src_img.cols;
            box.y = letter_box ? (float)(row[1] * 2 - 0.5 + anchor[2]) / anchor[3] * (float)IMAGE_HEIGHT * ratio : (sigmoid(row[1]) + anchor[2]) / anchor[3] * (float)src_img.rows;
            box.w = letter_box ? (float)pow(row[2] * 2, 2) * anchor[4] * ratio : exp(row[2]) * anchor[4] / (float)IMAGE_WIDTH * (float)src_img.cols;
            box.h = letter_box ? (float)pow(row[3] * 2, 2) * anchor[5] * ratio : exp(row[3]) * anchor[5] / (float)IMAGE_HEIGHT * (float)src_img.rows;
            result.det_results.push_back(box);
        }
        NmsDetect(result.det_results);
        vec_result.push_back(result);
        index++;
    }
    return vec_result;
}

float YOLOv4::sigmoid(float in){
    return 1.f / (1.f + exp(-in));
}
