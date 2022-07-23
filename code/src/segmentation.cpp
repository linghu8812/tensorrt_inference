//
// Created by linghu8812 on 2022/7/23.
//

#include "segmentation.h"

Segmentation::Segmentation(const YAML::Node &config) : Model(config) {
    OUT_WIDTH = config["OUT_WIDTH"].as<int>();
    OUT_HEIGHT = config["OUT_HEIGHT"].as<int>();
    CATEGORY = config["CATEGORY"].as<int>();
    class_colors.resize(CATEGORY);
    srand((int) time(nullptr));
    for (cv::Scalar &class_color : class_colors)
        class_color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
}

std::vector<SegmentationRes> Segmentation::InferenceImages(std::vector<cv::Mat> &vec_img) {
    auto t_start_pre = std::chrono::high_resolution_clock::now();
    std::vector<float> image_data = PreProcess(vec_img);
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
    std::cout << "segmentation prepare image take: " << total_pre << " ms." << std::endl;
    auto *output = new float[outSize * BATCH_SIZE];;
    auto t_start = std::chrono::high_resolution_clock::now();
    ModelInference(image_data, output);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "segmentation inference take: " << total_inf << " ms." << std::endl;
    auto r_start = std::chrono::high_resolution_clock::now();
    auto results = PostProcess(vec_img, output);
    auto r_end = std::chrono::high_resolution_clock::now();
    float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
    std::cout << "segmentation postprocess take: " << total_res << " ms." << std::endl;
    delete[] output;
    return results;
}

void Segmentation::InferenceFolder(const std::string &folder_name) {
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
            auto det_results = InferenceImages(vec_Mat);
            auto end_time = std::chrono::high_resolution_clock::now();
            DrawResults(det_results, vec_Mat, vec_name);
            vec_Mat = std::vector<cv::Mat>(BATCH_SIZE);
            batch_id = 0;
            total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count();
        }
    }
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
}

void Segmentation::DrawResults(const std::vector<SegmentationRes> &segmentations, std::vector<cv::Mat> &vec_img,
                               std::vector<std::string> image_name) {
    for (int i = 0; i < (int) vec_img.size(); i++) {
        auto org_img = vec_img[i];
        if (!org_img.data)
            continue;
        auto result = segmentations[i];
        if (channel_order == "BGR")
            cv::cvtColor(org_img, org_img, cv::COLOR_BGR2RGB);
        cv::Mat rsz_mat;
        cv::resize(result.seg_result, rsz_mat, cv::Size(org_img.cols, org_img.rows));
        std::vector<cv::Mat> vec_rst = {
                cv::Mat(cv::Size(org_img.cols, org_img.rows), CV_8UC1),
                cv::Mat(cv::Size(org_img.cols, org_img.rows), CV_8UC1),
                cv::Mat(cv::Size(org_img.cols, org_img.rows), CV_8UC1)
        };
        for (int h = 0; h < org_img.rows; h++) {
            auto *p = rsz_mat.ptr<uchar>(h);
            auto *r0 = vec_rst[0].ptr<uchar>(h);
            auto *r1 = vec_rst[1].ptr<uchar>(h);
            auto *r2 = vec_rst[2].ptr<uchar>(h);
            for (int w = 0; w < org_img.cols; w++) {
                r0[w] = p[w] == 0 ? 0 : class_colors[p[w] - 1][0];
                r1[w] = p[w] == 0 ? 0 : class_colors[p[w] - 1][1];
                r2[w] = p[w] == 0 ? 0 : class_colors[p[w] - 1][2];
            }
        }
        cv::Mat rst_mat;
        cv::merge(vec_rst, rst_mat);
        int pos = image_name[i].find_last_of(".");
        std::string rst_name = image_name[i].insert(pos, "_");
        std::cout << rst_name << std::endl;
        cv::imwrite(rst_name, rst_mat);
    }
}
