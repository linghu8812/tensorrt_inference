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
    feature_sizes = config["feature_sizes"].as<std::vector<int>>();
    feature_steps = config["feature_steps"].as<std::vector<int>>();
    feature_maps = config["feature_maps"].as<std::vector<int>>();
    anchor_sizes = config["anchor_sizes"].as<std::vector<std::vector<int>>>();
    int sum_of_feature = std::accumulate(feature_sizes.begin(), feature_sizes.end(), 0);
    out1_step = sum_of_feature * anchor_num;
    out2_step = sum_of_feature * anchor_num * bbox_head;
    out3_step = sum_of_feature * anchor_num * landmark_head;
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
    assert(engine->getNbBindings() == 4);
    void *buffers[4];
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

    int outSize[] = {
            int(bufferSize[1] / sizeof(float) / BATCH_SIZE),
            int(bufferSize[2] / sizeof(float) / BATCH_SIZE),
            int(bufferSize[3] / sizeof(float) / BATCH_SIZE)
    };

    EngineInference(sample_images, outSize, buffers, bufferSize, stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    // destroy the engine
    context->destroy();
    engine->destroy();
}

void RetinaFace::EngineInference(const std::vector<std::string> &image_list, const int *outSize, void **buffers,
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
            cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
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
            float out_1[outSize[0] * BATCH_SIZE];
            float out_2[outSize[1] * BATCH_SIZE];
            float out_3[outSize[2] * BATCH_SIZE];
            cudaMemcpyAsync(out_1, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(out_2, buffers[2], bufferSize[2], cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(out_3, buffers[3], bufferSize[3], cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            auto faces = postProcess(vec_Mat, out_1, out_2, out_3, outSize[0], outSize[1], outSize[3]);
            auto r_end = std::chrono::high_resolution_clock::now();
            float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
            std::cout << "Post process take: " << total_res << " ms." << std::endl;
            for (int i = 0; i < (int)vec_Mat.size(); i++)
            {
                auto org_img = vec_Mat[i];
                auto rects = faces[i];
                cv::cvtColor(org_img, org_img, cv::COLOR_BGR2RGB);
                for(const auto &rect : rects)
                {
                    char name[256];
                    sprintf(name, "%.2f", rect.confidence);
                    cv::putText(org_img, name, cv::Point(rect.face_box.x - rect.face_box.w / 2,rect.face_box.y - rect.face_box.h / 2 - 5),
                            cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
                    cv::Rect box(rect.face_box.x - rect.face_box.w / 2, rect.face_box.y - rect.face_box.h / 2, rect.face_box.w, rect.face_box.h);
                    cv::rectangle(org_img, box, cv::Scalar(255, 0, 0), 2, cv::LINE_8, 0);
                    for (int k = 0; k < rect.keypoints.size(); k++)
                    {
                        cv::Point2f key_point = rect.keypoints[k];
                        if (k % 3 == 0)
                            cv::circle(org_img, key_point, 3, cv::Scalar(0, 255, 0), -1);
                        else if (k % 3 == 1)
                            cv::circle(org_img, key_point, 3, cv::Scalar(0, 0, 255), -1);
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
}

void RetinaFace::GenerateAnchors() {
    float base_cx = 7.5;
    float base_cy = 7.5;

    priorbox_matrix = Eigen::MatrixXf(bbox_head, out1_step);
    int line = 0;
    for(size_t feature_map = 0; feature_map < feature_maps.size(); feature_map++) {
        for (int height = 0; height < feature_maps[feature_map]; ++height) {
            for (int width = 0; width < feature_maps[feature_map]; ++width) {
                for (int anchor = 0; anchor < anchor_sizes[feature_map].size(); ++anchor) {
                    float anchor_min_size = anchor_sizes[feature_map][anchor];
                    float cx = base_cx + width * feature_steps[feature_map];
                    float cy = base_cy + height * feature_steps[feature_map];
                    priorbox_matrix.col(line) << cx, cy, anchor_min_size, anchor_min_size;
                    line++;
                }
            }
        }
    }
}

std::vector<float> RetinaFace::prepareImage(std::vector<cv::Mat> &vec_img) {
    std::vector<float> result(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNEL);
    float *data = result.data();
    for (const cv::Mat &src_img : vec_img)
    {
        if (!src_img.data)
            continue;
        cv::Mat flt_img;
        cv::resize(src_img, flt_img, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
        flt_img.convertTo(flt_img, CV_32FC3);

        //HWC TO CHW
        std::vector<cv::Mat> split_img(INPUT_CHANNEL);
        cv::split(flt_img, split_img);

        int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT;
        for (int i = 0; i < INPUT_CHANNEL; ++i)
        {
            memcpy(data, split_img[i].data, channelLength * sizeof(float));
            data += channelLength;
        }
    }
    return result;
}

std::vector<std::vector<RetinaFace::FaceRes>> RetinaFace::postProcess(const std::vector<cv::Mat> &vec_Mat, float *output_1,
        float *output_2, float *output_3, const int &outSize_1, const int &outSize_2, const int &outSize_3) {
    std::vector<std::vector<FaceRes>> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat)
    {
        std::vector<FaceRes> result;
        float *out_1 = output_1 + index * outSize_1;
        float *out_2 = output_2 + index * outSize_2;
        float *out_3 = output_3 + index * outSize_3;
        float width_scale  = (float)src_img.cols / (float)IMAGE_WIDTH;
        float height_scale = (float)src_img.rows / (float)IMAGE_HEIGHT;

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Eigen::Unaligned > score_matrix(
                out_1, 1, out1_step);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Eigen::Unaligned > bbox_matrix(
                out_2, bbox_head, out2_step / bbox_head);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, Eigen::Unaligned > landmark_matrix(
                out_3, landmark_head, out3_step / landmark_head);

        auto score = score_matrix.row(0);
        for (int item = 0; item < score.cols(); ++item) {
            if(score[item] > obj_threshold){
                FaceRes headbox;
                headbox.confidence = score[item];

                headbox.face_box.x = (priorbox_matrix(0,item) + bbox_matrix(0,item) * priorbox_matrix(2,item)) * width_scale;
                headbox.face_box.y = (priorbox_matrix(1,item) + bbox_matrix(1,item) * priorbox_matrix(3,item)) * height_scale;
                headbox.face_box.w = priorbox_matrix(2,item) * exp(bbox_matrix(2,item)) * width_scale;
                headbox.face_box.h = priorbox_matrix(3,item) * exp(bbox_matrix(3,item)) * height_scale;

                headbox.keypoints = {
                        cv::Point2f((priorbox_matrix(0,item) + landmark_matrix(0, item) * priorbox_matrix(2,item)) * width_scale,
                                    (priorbox_matrix(1,item) + landmark_matrix(1, item) * priorbox_matrix(3,item)) * height_scale),
                        cv::Point2f((priorbox_matrix(0,item) + landmark_matrix(2, item) * priorbox_matrix(2,item)) * width_scale,
                                    (priorbox_matrix(1,item) + landmark_matrix(3, item) * priorbox_matrix(3,item)) * height_scale),
                        cv::Point2f((priorbox_matrix(0,item) + landmark_matrix(4, item) * priorbox_matrix(2,item)) * width_scale,
                                    (priorbox_matrix(1,item) + landmark_matrix(5, item) * priorbox_matrix(3,item)) * height_scale),
                        cv::Point2f((priorbox_matrix(0,item) + landmark_matrix(6, item) * priorbox_matrix(2,item)) * width_scale,
                                    (priorbox_matrix(1,item) + landmark_matrix(7, item) * priorbox_matrix(3,item)) * height_scale),
                        cv::Point2f((priorbox_matrix(0,item) + landmark_matrix(8, item) * priorbox_matrix(2,item)) * width_scale,
                                    (priorbox_matrix(1,item) + landmark_matrix(9, item) * priorbox_matrix(3,item)) * height_scale)
                };
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
