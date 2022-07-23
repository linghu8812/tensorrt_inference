#include "faces.h"

Faces::Faces(const YAML::Node &config) : Model(config) {
    obj_threshold = config["obj_threshold"].as<float>();
    nms_threshold = config["nms_threshold"].as<float>();
    detect_mask = config["detect_mask"].as<bool>();
    mask_thresh = config["mask_thresh"].as<float>();
    landmark_std = config["landmark_std"].as<float>();
    feature_steps = config["feature_steps"].as<std::vector<int>>();
    for (const int step:feature_steps) {
        assert(step != 0);
        int feature_height = IMAGE_HEIGHT / step;
        int feature_width = IMAGE_WIDTH / step;
        std::vector<int> feature_map = { feature_height, feature_width };
        feature_maps.push_back(feature_map);
        int feature_size = feature_height * feature_width;
        feature_sizes.push_back(feature_size);
    }
}

std::vector<FacesRes> Faces::InferenceImages(std::vector<cv::Mat> &vec_img) {
    auto t_start_pre = std::chrono::high_resolution_clock::now();
    std::vector<float> image_data = PreProcess(vec_img);
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
    std::cout << "face detection prepare image take: " << total_pre << " ms." << std::endl;
    auto *output = new float[outSize * BATCH_SIZE];;
    auto t_start = std::chrono::high_resolution_clock::now();
    ModelInference(image_data, output);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "face detection inference take: " << total_inf << " ms." << std::endl;
    auto r_start = std::chrono::high_resolution_clock::now();
    auto results = PostProcess(vec_img, output);
    auto r_end = std::chrono::high_resolution_clock::now();
    float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
    std::cout << "face detection postprocess take: " << total_res << " ms." << std::endl;
    delete[] output;
    return results;
}

void Faces::InferenceFolder(const std::string &folder_name) {
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

void Faces::NmsDetect(std::vector<FaceBox> &detections) {
    sort(detections.begin(), detections.end(), [=](const FaceBox &left, const FaceBox &right) {
        return left.bbox.prob > right.bbox.prob;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            float iou = Detection::IOUCalculate(detections[i].bbox, detections[j].bbox);
            if (iou > nms_threshold)
                detections[j].bbox.prob = 0;
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const FaceBox &det)
    { return det.bbox.prob == 0; }), detections.end());
}

void Faces::DrawResults(const std::vector<FacesRes> &faces, std::vector<cv::Mat> &vec_img,
                            std::vector<std::string> image_name=std::vector<std::string>()) {
    for (int i = 0; i < (int)vec_img.size(); i++) {
        auto org_img = vec_img[i];
        if (!org_img.data)
            continue;
        auto rects = faces[i].faces_results;
        if (channel_order == "BGR")
            cv::cvtColor(org_img, org_img, cv::COLOR_BGR2RGB);
        for(const auto &rect : rects) {
            char name[256];
            cv::Scalar color;
            sprintf(name, "%.2f", rect.bbox.prob);
            if (rect.has_mask) {
                color = cv::Scalar(0, 0, 255);
                cv::putText(org_img, "mask", cv::Point(rect.bbox.x,rect.bbox.y + 15),
                            cv::FONT_HERSHEY_COMPLEX, 0.7, color, 2);
            } else {
                color = cv::Scalar(255, 0, 0);
            }
            cv::putText(org_img, name, cv::Point(rect.bbox.x,rect.bbox.y - 5),
                        cv::FONT_HERSHEY_COMPLEX, 0.7, color, 2);
            cv::Rect box(rect.bbox.x, rect.bbox.y, rect.bbox.w, rect.bbox.h);
            cv::rectangle(org_img, box, color, 2, cv::LINE_8, 0);
            for (int k = 0; k < (int)rect.key_points.key_points.size(); k++) {
                KeyPoint key_point = rect.key_points.key_points[k];
                if (k % 3 == 0)
                    cv::circle(org_img, cv::Point(key_point.x, key_point.y), 3, cv::Scalar(0, 255, 0), -1);
                else if (k % 3 == 1)
                    cv::circle(org_img, cv::Point(key_point.x, key_point.y), 3, cv::Scalar(255, 0, 255), -1);
                else
                    cv::circle(org_img, cv::Point(key_point.x, key_point.y), 3, cv::Scalar(0, 255, 255), -1);
            }
        }
        int pos = image_name[i].find_last_of('.');
        std::string rst_name = image_name[i].insert(pos, "_");
        std::cout << rst_name << std::endl;
        cv::imwrite(rst_name, org_img);
    }
}