//
// Created by linghu8812 on 2022/7/18.
//

#include "feature.h"

Feature::Feature(const YAML::Node &config) : Model(config) {
}

std::vector<FeatureRes> Feature::InferenceImages(std::vector<cv::Mat> &vec_img) {
    auto t_start_pre = std::chrono::high_resolution_clock::now();
    std::vector<float> image_data = PreProcess(vec_img);
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
    std::cout << "extract feature prepare image take: " << total_pre << " ms." << std::endl;
    auto *output = new float[outSize * BATCH_SIZE];;
    auto t_start = std::chrono::high_resolution_clock::now();
    ModelInference(image_data, output);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "extract feature inference take: " << total_inf << " ms." << std::endl;
    auto r_start = std::chrono::high_resolution_clock::now();
    auto results = PostProcess(vec_img, output);
    auto r_end = std::chrono::high_resolution_clock::now();
    float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
    std::cout << "extract feature postprocess take: " << total_res << " ms." << std::endl;
    delete[] output;
    return results;
}

void Feature::InferenceFolder(const std::string &folder_name) {
    std::vector<std::string> image_list = ReadFolder(folder_name);
    int index = 0;
    int batch_id = 0;
    std::vector<cv::Mat> vec_Mat(BATCH_SIZE);
    std::vector<std::string> vec_name(BATCH_SIZE);
    std::vector<FeatureRes> vec_features;
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
            auto fet_results = InferenceImages(vec_Mat);
            auto end_time = std::chrono::high_resolution_clock::now();
            vec_features.insert(vec_features.end(), fet_results.begin(), fet_results.end());
            vec_Mat = std::vector<cv::Mat>(BATCH_SIZE);
            batch_id = 0;
            total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count();
        }
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    ComputeSimilarity(vec_features, vec_features);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "compute similarity take: " << std::chrono::duration<float, std::milli>(end_time - start_time).count() << " ms." << std::endl;
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
}

std::vector<FeatureRes> Feature::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<FeatureRes> vec_result;
    for (int i = 0; i < (int)vec_Mat.size(); i++)
    {
        FeatureRes result;
        cv::Mat onefeature(1, outSize, CV_32FC1, output + i * outSize);
        cv::normalize(onefeature, onefeature);
        result.feature = std::vector<float>(onefeature.reshape(1, outSize));
        vec_result.push_back(result);
    }
    return vec_result;
}

cv::Mat Feature::Feature2Mat(const std::vector<FeatureRes> &vec_results) {
    int index = 0;
    cv::Mat res_mat(vec_results.size(), outSize, CV_32FC1);
    for (const auto &result : vec_results) {
        cv::Mat mat = cv::Mat(result.feature).clone().reshape(1, outSize).t();
        mat.copyTo(res_mat.rowRange(index, index + 1));
        index++;
    }
    return res_mat;
}

void Feature::ComputeSimilarity(const std::vector<FeatureRes> &results_a, const std::vector<FeatureRes> &results_b) {
    cv::Mat mat_a = Feature2Mat(results_a);
    cv::Mat mat_b = Feature2Mat(results_b);
    cv::Mat similarity = mat_a * mat_b.t();
    std::cout << "The similarity matrix of the image folder is:\n" << (similarity + 1) / 2 << "!" << std::endl;
}