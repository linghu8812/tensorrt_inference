#include "nanodet.h"
#include "yaml-cpp/yaml.h"
#include "common.hpp"


inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{ 0 };
    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }
    return 0;
}

nanodet::nanodet(const std::string &config_file) {
    YAML::Node root = YAML::LoadFile(config_file);
    YAML::Node config = root["nanodet"];
    onnx_file = config["onnx_file"].as<std::string>();
    engine_file = config["engine_file"].as<std::string>();
    labels_file = config["labels_file"].as<std::string>();
    BATCH_SIZE = config["BATCH_SIZE"].as<int>();
    INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
    IMAGE_WIDTH = config["IMAGE_WIDTH"].as<int>();
    IMAGE_HEIGHT = config["IMAGE_HEIGHT"].as<int>();
    img_mean = config["img_mean"].as<std::vector<float>>();
    img_std = config["img_mean"].as<std::vector<float>>();
    obj_threshold = config["obj_threshold"].as<float>();
    nms_threshold = config["nms_threshold"].as<float>();
    strides = config["strides"].as<std::vector<int>>();
    reg_max = config["reg_max"].as<int>() + 1;
    detect_labels = readCOCOLabel(labels_file);
    CATEGORY = detect_labels.size();
    class_colors.resize(CATEGORY);
    GenerateReferMatrix();
    srand((int) time(nullptr));
    for (cv::Scalar &class_color : class_colors)
        class_color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
    float project[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    cv::Mat project_mat = cv::Mat(reg_max, 1, CV_32FC1, project);
}

nanodet::~nanodet() = default;

void nanodet::LoadEngine() {
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

bool nanodet::InferenceFolder(const std::string &folder_name) {
    std::vector<std::string> sample_images = readFolder(folder_name);
    //get context
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);

    //get buffers
    assert(engine->getNbBindings() == 7);
    void *buffers[7];
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
            int(bufferSize[3] / sizeof(float) / BATCH_SIZE),
            int(bufferSize[4] / sizeof(float) / BATCH_SIZE),
            int(bufferSize[5] / sizeof(float) / BATCH_SIZE),
            int(bufferSize[6] / sizeof(float) / BATCH_SIZE)
    };

    EngineInference(sample_images, outSize, buffers, bufferSize, stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);
    cudaFree(buffers[3]);
    cudaFree(buffers[4]);
    cudaFree(buffers[5]);
    cudaFree(buffers[6]);

    // destroy the engine
    context->destroy();
    engine->destroy();
}

void nanodet::EngineInference(const std::vector<std::string> &image_list, const int *outSize, void **buffers,
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
            auto *out_1 = new float[outSize[0] * BATCH_SIZE];
            auto *out_2 = new float[outSize[1] * BATCH_SIZE];
            auto *out_3 = new float[outSize[2] * BATCH_SIZE];
            auto *out_4 = new float[outSize[3] * BATCH_SIZE];
            auto *out_5 = new float[outSize[4] * BATCH_SIZE];
            auto *out_6 = new float[outSize[5] * BATCH_SIZE];
            cudaMemcpyAsync(out_1, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(out_2, buffers[2], bufferSize[2], cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(out_3, buffers[3], bufferSize[3], cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(out_4, buffers[4], bufferSize[4], cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(out_5, buffers[5], bufferSize[5], cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(out_6, buffers[6], bufferSize[6], cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            auto results = postProcess(vec_Mat, out_1, out_2, out_3, out_4, out_5, out_6,
                    outSize[0], outSize[1], outSize[2], outSize[3], outSize[4], outSize[5]);
            delete[] out_1;
            delete[] out_2;
            delete[] out_3;
            auto r_end = std::chrono::high_resolution_clock::now();
            float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
            std::cout << "Post process take: " << total_res << " ms." << std::endl;
            for (int i = 0; i < (int)vec_Mat.size(); i++)
            {
                auto org_img = vec_Mat[i];
                if (!org_img.data)
                    continue;
                auto rects = results[i];
//                cv::cvtColor(org_img, org_img, cv::COLOR_BGR2RGB);
                for(const auto &rect : rects)
                {
                    char t[256];
                    sprintf(t, "%.2f", rect.prob);
                    std::string name = detect_labels[rect.classes] + "-" + t;
                    cv::putText(org_img, name, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, class_colors[rect.classes], 2);
                    cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
                    cv::rectangle(org_img, rst, class_colors[rect.classes], 2, cv::LINE_8, 0);                }
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

void nanodet::GenerateReferMatrix() {
    anchor_mat.resize(strides.size());
    int index = 0;
    for (const int &stride : strides) {
        anchor_mat[index] = cv::Mat(IMAGE_WIDTH / stride * IMAGE_HEIGHT / stride, 2, CV_32FC1);
        for (int h = 0; h < IMAGE_HEIGHT / stride; h++)
            for (int w = 0; w < IMAGE_WIDTH / stride; w++) {
                auto *row = anchor_mat[index].ptr<float>(h * IMAGE_WIDTH / stride + w);
                row[0] = float((2 * w + 1) * stride - 1) / 2;
                row[1] = float((2 * h + 1) * stride - 1) / 2;
            }
        index += 1;
    }
}

std::vector<float> nanodet::prepareImage(std::vector<cv::Mat> &vec_img) {
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
                cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1, data + channelLength * (index + 2)),
                cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1, data + channelLength * (index + 1)),
                cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1, data + channelLength * index)
        };
        index += 3;
        cv::split(flt_img, split_img);
        for (int i = 0; i < INPUT_CHANNEL; i++) {
            split_img[i] = (split_img[i] - img_mean[i]) / img_std[i];
        }
    }
    return result;
}

void nanodet::decode_boxes(float *out, float *box, std::vector<nanodet::DetectRes> &result, const float &ratio, const int &outSize,
        const int &boxSize, const int &stride, const cv::Mat &anchor) {
    int rows = outSize / CATEGORY;
    cv::Mat score_mat = cv::Mat(rows, CATEGORY, CV_32FC1, out);
    cv::Mat boxes_mat = cv::Mat(rows, boxSize / rows, CV_32FC1, box);
    cv::Mat reshape_mat = boxes_mat.reshape(0, boxes_mat.rows * 4);
    float project[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    cv::Mat project_mat = cv::Mat(8, 1, CV_32FC1, project);
    for (int row = 0; row < rows * 4; row++) {
        auto o_p = reshape_mat.ptr<float>(row);
        activation_function_softmax(o_p, o_p, reshape_mat.cols);
    }
    cv::Mat box_result = reshape_mat * project_mat;
    box_result = box_result.reshape(0, box_result.rows / 4) * stride;
    box_result.col(0) = anchor.col(0) - box_result.col(0);
    box_result.col(1) = anchor.col(1) - box_result.col(1);
    box_result.col(2) = anchor.col(0) + box_result.col(2);
    box_result.col(3) = anchor.col(1) + box_result.col(3);
    for (int row = 0; row < rows; row++) {
        DetectRes box;
        auto score = score_mat.ptr<float>(row);
        auto max_pos = std::max_element(score, score + CATEGORY);
        box.prob = score[max_pos - score];
        if (box.prob < obj_threshold)
            continue;
        box.classes = max_pos - score;
        auto rst = box_result.ptr<float>(row);
        box.x = (rst[0] + rst[2]) / 2 * ratio;
        box.y = (rst[1] + rst[3]) / 2 * ratio;
        box.w = (rst[2] - rst[0]) * ratio;
        box.h = (rst[3] - rst[1]) * ratio;
        result.push_back(box);
    }
    int a = 0;
}

std::vector<std::vector<nanodet::DetectRes>> nanodet::postProcess(const std::vector<cv::Mat> &vec_Mat,
        float *output_1, float *output_2, float *output_3, float *output_4, float *output_5, float *output_6,
        const int &outSize_1, const int &outSize_2, const int &outSize_3, const int &outSize_4, const int &outSize_5, const int &outSize_6) {
    std::vector<std::vector<DetectRes>> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat)
    {
        std::vector<DetectRes> result, result1, result2, result3;
        float ratio = float(src_img.cols) / float(IMAGE_WIDTH) > float(src_img.rows) / float(IMAGE_HEIGHT)  ? float(src_img.cols) / float(IMAGE_WIDTH) : float(src_img.rows) / float(IMAGE_HEIGHT);
        float *box1 = output_1 + index * outSize_1;
        float *out1 = output_2 + index * outSize_2;
        float *box2 = output_3 + index * outSize_3;
        float *out2 = output_4 + index * outSize_4;
        float *out3 = output_5 + index * outSize_5;
        float *box3 = output_6 + index * outSize_6;
        decode_boxes(out1, box1, result1, ratio, outSize_2, outSize_1, strides[0], anchor_mat[0]);
        decode_boxes(out2, box2, result2, ratio, outSize_4, outSize_3, strides[1], anchor_mat[1]);
        decode_boxes(out3, box3, result3, ratio, outSize_5, outSize_6, strides[2], anchor_mat[2]);
        result.insert(result.end(), result1.begin(), result1.end());
        result.insert(result.end(), result2.begin(), result2.end());
        result.insert(result.end(), result3.begin(), result3.end());
        NmsDetect(result);
        vec_result.push_back(result);
        index++;
    }
    return vec_result;
}

void nanodet::NmsDetect(std::vector<DetectRes> &detections) {
    sort(detections.begin(), detections.end(), [=](const DetectRes &left, const DetectRes &right) {
        return left.prob > right.prob;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            float iou = IOUCalculate(detections[i], detections[j]);
            if (iou > nms_threshold)
                detections[j].prob = 0;
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const DetectRes &det)
    { return det.prob == 0; }), detections.end());
}

float nanodet::IOUCalculate(const nanodet::DetectRes &det_a, const nanodet::DetectRes &det_b) {
    cv::Point2f center_a(det_a.x, det_a.y);
    cv::Point2f center_b(det_b.x, det_b.y);
    cv::Point2f left_up(std::min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
                        std::min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
    cv::Point2f right_down(std::max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
                           std::max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2 ? det_a.x - det_a.w / 2 : det_b.x - det_b.w / 2;
    float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2 ? det_a.y - det_a.h / 2 : det_b.y - det_b.h / 2;
    float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2 ? det_a.x + det_a.w / 2 : det_b.x + det_b.w / 2;
    float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2 ? det_a.y + det_a.h / 2 : det_b.y + det_b.h / 2;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        return inter_area / union_area - distance_d / distance_c;
}
