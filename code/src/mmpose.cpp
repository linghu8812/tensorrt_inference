#include "mmpose.h"
#include <cmath>

mmpose::mmpose(const YAML::Node &config) : KeyPoints(config) {
    num_key_points = config["num_key_points"].as<int>();
    skeleton = config["skeleton"].as<std::vector<std::vector<int>>>();
    point_thresh = config["point_thresh"].as<float>();
}

std::vector<mmposeRes> mmpose::InferenceImages(std::vector<cv::Mat> &vec_img) {
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

void mmpose::InferenceFolder(const std::string &folder_name) {
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

std::vector<mmposeRes> mmpose::PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output) {
    std::vector<mmposeRes> vec_key_points;
    int feature_size = IMAGE_WIDTH * IMAGE_HEIGHT / 16;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat) {
        mmposeRes key_points;
        key_points.key_points = std::vector<mmposePoint>(num_key_points);
        float ratio = std::max(float(src_img.cols) / float(IMAGE_WIDTH), float(src_img.rows) / float(IMAGE_HEIGHT));
        float *current_person = output + index * outSize;
        for (int number = 0; number < num_key_points; number++) {
            float *current_point = current_person + feature_size * number;
            auto max_pos = std::max_element(current_point, current_point + feature_size);
            key_points.key_points[number].prob = *max_pos;
            
            float *end_point = current_point + feature_size - 1;
            
            float *minx = std::min(max_pos + 1, end_point);
            float *maxx = std::max(max_pos - 1, current_point);
            float x = (float)((max_pos - current_point) % (IMAGE_WIDTH / 4)) +
                    (*(minx) > *(maxx) ? 0.25f : -0.25f);
            
            float *miny = std::min(end_point, (max_pos + IMAGE_WIDTH / 4));
            float *maxy = std::max(max_pos - IMAGE_WIDTH / 4, current_point);
            float y = (float)(max_pos - current_point) / ((float)IMAGE_WIDTH / 4.0f) +
                    (*miny > *maxy ? 0.25f : -0.25f);
            
            key_points.key_points[number].x = std::round(x * ratio * 4);
            key_points.key_points[number].y = std::round(y * ratio * 4);
            key_points.key_points[number].number = number;
        }
        vec_key_points.push_back(key_points);
        index++;
    }
    return vec_key_points;
}

void mmpose::DrawResults(const std::vector<mmposeRes> &results, std::vector<cv::Mat> &vec_img,
                         std::vector<std::string> image_names) {
    for (int i = 0; i < (int)vec_img.size(); i++) {
        auto org_img = vec_img[i];
        if (!org_img.data)
            continue;
        auto current_points = results[i].key_points;
        if (channel_order == "BGR")
            cv::cvtColor(org_img, org_img, cv::COLOR_BGR2RGB);
        for (const auto &bone : skeleton) {
            if (current_points[bone[0]].prob < point_thresh or current_points[bone[1]].prob < point_thresh)
                continue;
            cv::Scalar color;
            if (bone[0] < 5 or bone[1] < 5)
                color = cv::Scalar(0, 255, 0);
            else if (bone[0] > 12 or bone[1] > 12)
                color = cv::Scalar(255, 0, 0);
            else if (bone[0] > 4 and bone[0] < 11 and bone[1] > 4 and bone[1] < 11)
                color = cv::Scalar(0, 255, 255);
            else
                color = cv::Scalar(255, 0, 255);
            cv::line(org_img, cv::Point(current_points[bone[0]].x, current_points[bone[0]].y),
                     cv::Point(current_points[bone[1]].x, current_points[bone[1]].y), color,
                     2);
        }
        for(const auto &point : current_points) {
            if (point.prob < point_thresh)
                continue;
            cv::Scalar color;
            if (point.number < 5)
                color = cv::Scalar(0, 255, 0);
            else if (point.number > 10)
                color = cv::Scalar(255, 0, 0);
            else
                color = cv::Scalar(0, 255, 255);
            cv::circle(org_img, cv::Point(point.x, point.y), 5, color, -1, cv::LINE_8, 0);
        }
        int pos = image_names[i].find_last_of('.');
        std::string rst_name = image_names[i].insert(pos, "_");
        std::cout << rst_name << std::endl;
        cv::imwrite(rst_name, org_img);
    }
}
