//
// Created by linghu8812 on 2022/7/19.
//

#include "keypoints.h"

KeyPoints::KeyPoints(const YAML::Node &config) : Model(config) {}

std::vector<KeyPointsRes> KeyPoints::InferenceImages(std::vector<cv::Mat> &vec_img) {
    auto t_start_pre = std::chrono::high_resolution_clock::now();
    std::vector<float> image_data = PreProcess(vec_img);
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
    std::cout << "extract key points prepare image take: " << total_pre << " ms." << std::endl;
    auto *output = new float[outSize * BATCH_SIZE];;
    auto t_start = std::chrono::high_resolution_clock::now();
    ModelInference(image_data, output);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "extract key points inference take: " << total_inf << " ms." << std::endl;
    auto r_start = std::chrono::high_resolution_clock::now();
    auto results = PostProcess(vec_img, output);
    auto r_end = std::chrono::high_resolution_clock::now();
    float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
    std::cout << "extract key points postprocess take: " << total_res << " ms." << std::endl;
    delete[] output;
    return results;
}

void KeyPoints::InferenceFolder(const std::string &folder_name) {
    std::vector<std::string> image_list = ReadFolder(folder_name);
    int index = 0;
    int batch_id = 0;
    std::vector<cv::Mat> vec_Mat(BATCH_SIZE);
    std::vector<std::string> vec_name(BATCH_SIZE);
    float total_time = 0;
    for (const std::string &image_name : image_list) {
        index++;
        std::cout << "Processing: " << image_name << std::endl;
        cv::Mat src_img = cv::imread(image_name);
        if (src_img.data) {
            if (channel_order == "BGR")
                cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
            vec_Mat[batch_id] = src_img.clone();
            vec_name[batch_id] = image_name;
            batch_id++;
        }
        if (batch_id == BATCH_SIZE or index == image_list.size()) {
            auto start_time = std::chrono::high_resolution_clock::now();
            auto kpt_results = InferenceImages(vec_Mat);
            auto end_time = std::chrono::high_resolution_clock::now();
            DrawResults(kpt_results, vec_Mat, vec_name);
            vec_Mat = std::vector<cv::Mat>(BATCH_SIZE);
            batch_id = 0;
            total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count();
        }
    }
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
}

void KeyPoints::DrawResults(const std::vector<KeyPointsRes> &results, std::vector<cv::Mat> &vec_img,
                            std::vector<std::string> image_names) {
    for (int i = 0; i < (int)vec_img.size(); i++) {
        auto org_img = vec_img[i];
        if (!org_img.data)
            continue;
        auto points = results[i].key_points;
        if (channel_order == "BGR")
            cv::cvtColor(org_img, org_img, cv::COLOR_BGR2RGB);
        for(const auto &point : points) {
            cv::circle(org_img, cv::Point(point.x, point.y), 1, cv::Scalar(200, 160, 75), -1, cv::LINE_8, 0);
        }
        int pos = image_names[i].find_last_of('.');
        std::string rst_name = image_names[i].insert(pos, "_");
        std::cout << rst_name << std::endl;
        cv::imwrite(rst_name, org_img);
    }
}

std::vector<KeyPointsRes> KeyPoints::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<KeyPointsRes> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat) {
        KeyPointsRes result;
        float *out = output + index * outSize;
        cv::Mat result_matrix = cv::Mat(1, outSize, CV_32FC1, out);
        result_matrix = (result_matrix + 1) * src_img.cols / 2;
        for (int start = 0; start < outSize; start += 2) {
            KeyPoint point = {int(out[start]), int(out[start + 1])};
            result.key_points.push_back(point);
        }
        vec_result.push_back(result);
        index++;
    }
    return vec_result;
}