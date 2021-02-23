#include "RetinaFace.h"
#include "yaml-cpp/yaml.h"
#include "common.hpp"

RetinaFace::RetinaFace(const std::string &config_file) {
    YAML::Node root = YAML::LoadFile(config_file);
    YAML::Node config = root["RetinaFace"];
    onnx_file = config["onnx_file"].as<std::string>();
    engine_file = config["engine_file"].as<std::string>();
    BATCH_SIZE = config["BATCH_SIZE"].as<int>();
    INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
    IMAGE_WIDTH = config["IMAGE_WIDTH"].as<int>();
    IMAGE_HEIGHT = config["IMAGE_HEIGHT"].as<int>();
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
    anchor_sizes = config["anchor_sizes"].as<std::vector<std::vector<int>>>();
    sum_of_feature = std::accumulate(feature_sizes.begin(), feature_sizes.end(), 0) * anchor_num;
    GenerateAnchors();
}

RetinaFace::~RetinaFace() = default;

void RetinaFace::LoadEngine() {
    // create and load engine
    std::fstream existEngine;
    existEngine.open(engine_file, std::ios::in);
    if (existEngine) {
        readTrtFile(engine_file, engine);
        assert(engine != nullptr);
    } else {
        onnxToTRTModel(onnx_file, engine_file, engine, BATCH_SIZE);
        assert(engine != nullptr);
    }
}

bool RetinaFace::InferenceFolder(const std::string &folder_name) {
    std::vector<std::string> sample_images = readFolder(folder_name);
    //get context
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);

    //get buffers
    assert(engine->getNbBindings() == 2);
    void *buffers[2];
    std::vector<int64_t> bufferSize;
    int nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);

    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        bufferSize[i] = totalSize;
        std::cout << "binding" << i << ": " << totalSize << std::endl;
        cudaMalloc(&buffers[i], totalSize);
    }

    //get stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int outSize = int(bufferSize[1] / sizeof(float) / BATCH_SIZE);

    EngineInference(sample_images, outSize, buffers, bufferSize, stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    // destroy the engine
    context->destroy();
    engine->destroy();
}

void RetinaFace::EngineInference(const std::vector<std::string> &image_list, const int &outSize, void **buffers,
                                 const std::vector<int64_t> &bufferSize, cudaStream_t stream) {
    int index = 0;
    int batch_id = 0;
    std::vector<cv::Mat> vec_Mat(BATCH_SIZE);
    std::vector<std::string> vec_name(BATCH_SIZE);
    cv::Mat face_feature(image_list.size(), outSize, CV_32FC1);
    float total_time = 0;
    for (const std::string &image_name : image_list)
    {
        index++;
        std::cout << "Processing: " << image_name << std::endl;
        cv::Mat src_img = cv::imread(image_name);
        if (src_img.data)
        {
//            cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
            vec_Mat[batch_id] = src_img.clone();
            vec_name[batch_id] = image_name;
            batch_id++;
        }
        if (batch_id == BATCH_SIZE or index == image_list.size())
        {
            auto t_start_pre = std::chrono::high_resolution_clock::now();
            std::cout << "prepareImage" << std::endl;
            std::vector<float>curInput = prepareImage(vec_Mat);
            auto t_end_pre = std::chrono::high_resolution_clock::now();
            float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
            std::cout << "prepare image take: " << total_pre << " ms." << std::endl;
            total_time += total_pre;
            batch_id = 0;
            if (!curInput.data()) {
                std::cout << "prepare images ERROR!" << std::endl;
                continue;
            }
            // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
            std::cout << "host2device" << std::endl;
            cudaMemcpyAsync(buffers[0], curInput.data(), bufferSize[0], cudaMemcpyHostToDevice, stream);

            // do inference
            std::cout << "execute" << std::endl;
            auto t_start = std::chrono::high_resolution_clock::now();
            context->execute(BATCH_SIZE, buffers);
            auto t_end = std::chrono::high_resolution_clock::now();
            float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            std::cout << "Inference take: " << total_inf << " ms." << std::endl;
            total_time += total_inf;
            std::cout << "execute success" << std::endl;
            std::cout << "device2host" << std::endl;
            std::cout << "post process" << std::endl;
            auto r_start = std::chrono::high_resolution_clock::now();
            auto *out = new float[outSize * BATCH_SIZE];
            cudaMemcpyAsync(out, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            auto faces = postProcess(vec_Mat, out, outSize);
            delete[] out;
            auto r_end = std::chrono::high_resolution_clock::now();
            float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
            std::cout << "Post process take: " << total_res << " ms." << std::endl;
            for (int i = 0; i < (int)vec_Mat.size(); i++)
            {
                auto org_img = vec_Mat[i];
                if (!org_img.data)
                    continue;
                auto rects = faces[i];
//                cv::cvtColor(org_img, org_img, cv::COLOR_BGR2RGB);
                for(const auto &rect : rects)
                {
                    char name[256];
                    cv::Scalar color;
                    sprintf(name, "%.2f", rect.confidence);
                    if (rect.has_mask) {
                        color = cv::Scalar(0, 0, 255);
                        cv::putText(org_img, "mask", cv::Point(rect.face_box.x - rect.face_box.w / 2,rect.face_box.y - rect.face_box.h / 2 + 15),
                                    cv::FONT_HERSHEY_COMPLEX, 0.7, color, 2);
                    } else {
                        color = cv::Scalar(255, 0, 0);
                    }
                    cv::putText(org_img, name, cv::Point(rect.face_box.x - rect.face_box.w / 2,rect.face_box.y - rect.face_box.h / 2 - 5),
                            cv::FONT_HERSHEY_COMPLEX, 0.7, color, 2);
                    cv::Rect box(rect.face_box.x - rect.face_box.w / 2, rect.face_box.y - rect.face_box.h / 2, rect.face_box.w, rect.face_box.h);
                    cv::rectangle(org_img, box, color, 2, cv::LINE_8, 0);
                    for (int k = 0; k < rect.keypoints.size(); k++)
                    {
                        cv::Point2f key_point = rect.keypoints[k];
                        if (k % 3 == 0)
                            cv::circle(org_img, key_point, 3, cv::Scalar(0, 255, 0), -1);
                        else if (k % 3 == 1)
                            cv::circle(org_img, key_point, 3, cv::Scalar(255, 0, 255), -1);
                        else
                            cv::circle(org_img, key_point, 3, cv::Scalar(0, 255, 255), -1);
                    }
                }
                int pos = vec_name[i].find_last_of(".");
                std::string rst_name = vec_name[i].insert(pos, "_");
                std::cout << rst_name << std::endl;
                cv::imwrite(rst_name, org_img);
            }
            total_time += total_res;
            vec_Mat = std::vector<cv::Mat>(BATCH_SIZE);
        }
    }
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
}

void RetinaFace::GenerateAnchors() {
    float base_cx = 7.5;
    float base_cy = 7.5;

    refer_matrix = cv::Mat(sum_of_feature, bbox_head, CV_32FC1);
    int line = 0;
    for(size_t feature_map = 0; feature_map < feature_maps.size(); feature_map++) {
        for (int height = 0; height < feature_maps[feature_map][0]; ++height) {
            for (int width = 0; width < feature_maps[feature_map][1]; ++width) {
                for (int anchor = 0; anchor < anchor_sizes[feature_map].size(); ++anchor) {
                    auto *row = refer_matrix.ptr<float>(line);
                    row[0] = base_cx + (float)width * feature_steps[feature_map];
                    row[1] = base_cy + (float)height * feature_steps[feature_map];
                    row[2] = anchor_sizes[feature_map][anchor];
                    line++;
                }
            }
        }
    }
}

std::vector<float> RetinaFace::prepareImage(std::vector<cv::Mat> &vec_img) {
    std::vector<float> result(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNEL);
    float *data = result.data();
    int index = 0;
    for (const cv::Mat &src_img : vec_img)
    {
        if (!src_img.data)
            continue;
        float ratio = float(IMAGE_WIDTH) / float(src_img.cols) < float(IMAGE_HEIGHT) / float(src_img.rows) ? float(IMAGE_WIDTH) / float(src_img.cols) : float(IMAGE_HEIGHT) / float(src_img.rows);
        cv::Mat flt_img = cv::Mat::zeros(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3);
        cv::Mat rsz_img;
        cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);
        rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
        flt_img.convertTo(flt_img, CV_32FC3);

        //HWC TO CHW
        int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT;
        std::vector<cv::Mat> split_img = {
                cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data + channelLength * (index + 2)),
                cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data + channelLength * (index + 1)),
                cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data + channelLength * index)
        };
        index += 3;
        cv::split(flt_img, split_img);
    }
    return result;
}

std::vector<std::vector<RetinaFace::FaceRes>> RetinaFace::postProcess(const std::vector<cv::Mat> &vec_Mat,
        float *output, const int &outSize) {
    std::vector<std::vector<FaceRes>> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat)
    {
        std::vector<FaceRes> result;
        float *out = output + index * outSize;
        float ratio = float(src_img.cols) / float(IMAGE_WIDTH) > float(src_img.rows) / float(IMAGE_HEIGHT)  ? float(src_img.cols) / float(IMAGE_WIDTH) : float(src_img.rows) / float(IMAGE_HEIGHT);

        int result_cols = (detect_mask ? 3 : 2) + bbox_head + landmark_head;
        cv::Mat result_matrix = cv::Mat(sum_of_feature, result_cols, CV_32FC1, out);

        for (int item = 0; item < result_matrix.rows; ++item) {
            auto *current_row = result_matrix.ptr<float>(item);
            if(current_row[0] > obj_threshold){
                FaceRes headbox;
                headbox.confidence = current_row[0];
                auto *anchor = refer_matrix.ptr<float>(item);
                auto *bbox = current_row + 1;
                auto *keyp = current_row + 2 + bbox_head;
                auto *mask = current_row + 2 + bbox_head + landmark_head;

                headbox.face_box.x = (anchor[0] + bbox[0] * anchor[2]) * ratio;
                headbox.face_box.y = (anchor[1] + bbox[1] * anchor[2]) * ratio;
                headbox.face_box.w = anchor[2] * exp(bbox[2]) * ratio;
                headbox.face_box.h = anchor[2] * exp(bbox[3]) * ratio;

                headbox.keypoints = {
                        cv::Point2f((anchor[0] + keyp[0] * anchor[2] * landmark_std) * ratio,
                                    (anchor[1] + keyp[1] * anchor[2] * landmark_std) * ratio),
                        cv::Point2f((anchor[0] + keyp[2] * anchor[2] * landmark_std) * ratio,
                                    (anchor[1] + keyp[3] * anchor[2] * landmark_std) * ratio),
                        cv::Point2f((anchor[0] + keyp[4] * anchor[2] * landmark_std) * ratio,
                                    (anchor[1] + keyp[5] * anchor[2] * landmark_std) * ratio),
                        cv::Point2f((anchor[0] + keyp[6] * anchor[2] * landmark_std) * ratio,
                                    (anchor[1] + keyp[7] * anchor[2] * landmark_std) * ratio),
                        cv::Point2f((anchor[0] + keyp[8] * anchor[2] * landmark_std) * ratio,
                                    (anchor[1] + keyp[9] * anchor[2] * landmark_std) * ratio)
                };

                if (detect_mask and mask[0] > mask_thresh)
                    headbox.has_mask = true;
                result.push_back(headbox);
            }
        }
        NmsDetect(result);
        vec_result.push_back(result);
        index++;
    }
    return vec_result;
}

void RetinaFace::NmsDetect(std::vector<FaceRes> &detections) {
    sort(detections.begin(), detections.end(), [=](const FaceRes &left, const FaceRes &right) {
        return left.confidence > right.confidence;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            float iou = IOUCalculate(detections[i].face_box, detections[j].face_box);
            if (iou > nms_threshold)
                detections[j].confidence = 0;
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const FaceRes &det)
    { return det.confidence == 0; }), detections.end());
}

float RetinaFace::IOUCalculate(const RetinaFace::FaceBox &det_a, const RetinaFace::FaceBox &det_b) {
    cv::Point2f center_a(det_a.x + det_a.w / 2, det_a.y + det_a.h / 2);
    cv::Point2f center_b(det_b.x + det_b.w / 2, det_b.y + det_b.h / 2);
    cv::Point2f left_up(std::min(det_a.x, det_b.x),std::min(det_a.y, det_b.y));
    cv::Point2f right_down(std::max(det_a.x + det_a.w, det_b.x + det_b.w),std::max(det_a.y + det_a.h, det_b.y + det_b.h));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x > det_b.x ? det_a.x : det_b.x;
    float inter_t = det_a.y > det_b.y ? det_a.y : det_b.y;
    float inter_r = det_a.x + det_a.w < det_b.x + det_b.w ? det_a.x + det_a.w : det_b.x + det_b.w;
    float inter_b = det_a.y + det_a.h < det_b.y + det_b.h ? det_a.y + det_a.h : det_b.y + det_b.h;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        return inter_area / union_area - distance_d / distance_c;
}
