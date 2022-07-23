#include "gender-age.h"

GenderAge::GenderAge(const YAML::Node &config) : Model(config) {}

std::vector<attribute> GenderAge::InferenceImages(std::vector<cv::Mat> &vec_img) {
    auto t_start_pre = std::chrono::high_resolution_clock::now();
    std::vector<float> image_data = PreProcess(vec_img);
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
    std::cout << "classification prepare image take: " << total_pre << " ms." << std::endl;
    auto *output = new float[outSize * BATCH_SIZE];;
    auto t_start = std::chrono::high_resolution_clock::now();
    ModelInference(image_data, output);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "classification inference take: " << total_inf << " ms." << std::endl;
    auto r_start = std::chrono::high_resolution_clock::now();
    auto results = PostProcess(vec_img, output);
    auto r_end = std::chrono::high_resolution_clock::now();
    float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
    std::cout << "classification postprocess take: " << total_res << " ms." << std::endl;
    delete[] output;
    return results;
}

void GenderAge::InferenceFolder(const std::string &folder_name) {
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
            auto gar_results = InferenceImages(vec_Mat);
            auto end_time = std::chrono::high_resolution_clock::now();
            DrawResults(gar_results, vec_Mat, vec_name);
            vec_Mat = std::vector<cv::Mat>(BATCH_SIZE);
            batch_id = 0;
            total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count();
        }
    }
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
}

void GenderAge::DrawResults(const std::vector<attribute> &results, std::vector<cv::Mat> &vec_img,
                            std::vector<std::string> image_names) {
    for (int i = 0; i < (int)vec_img.size(); i++) {
        auto org_img = vec_img[i];
        if (!org_img.data)
            continue;
        auto attribute = results[i];
        if (!image_names.empty()) {
            std::cout << image_names[i] << ": gender: " << attribute.gender << ": age: " << attribute.age << std::endl;
        }
    }
}

std::vector<attribute> GenderAge::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<attribute> results;
    for (int i = 0; i < (int)vec_Mat.size(); i++) {
        auto current_out = output + i * outSize;
        attribute result;
        result.gender = current_out[0] < current_out[1];
        result.age = 0;
        for (int j = 2; j < outSize; j+=2)
        {
            result.age += current_out[j] < current_out[j + 1];
        }
        results.push_back(result);
    }
    return results;
}
