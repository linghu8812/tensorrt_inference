#include "Yolov4.h"
#include "yaml-cpp/yaml.h"
#include "common.hpp"

YOLOv4::YOLOv4(const std::string &config_file) {
    YAML::Node root = YAML::LoadFile(config_file);
    YAML::Node config = root["Yolov4"];
    onnx_file = config["onnx_file"].as<std::string>();
    engine_file = config["engine_file"].as<std::string>();
    labels_file = config["labels_file"].as<std::string>();
    BATCH_SIZE = config["BATCH_SIZE"].as<int>();
    INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
    IMAGE_WIDTH = config["IMAGE_WIDTH"].as<int>();
    IMAGE_HEIGHT = config["IMAGE_HEIGHT"].as<int>();
    obj_threshold = config["obj_threshold"].as<float>();
    nms_threshold = config["nms_threshold"].as<float>();
    stride = config["stride"].as<std::vector<int>>();
    anchors = config["anchors"].as<std::vector<std::vector<int>>>();
    coco_labels = readCOCOLabel(labels_file);
    CATEGORY = coco_labels.size();
    grids = {
            {3, int(IMAGE_WIDTH / stride[0]), int(IMAGE_HEIGHT / stride[0])},
            {3, int(IMAGE_WIDTH / stride[1]), int(IMAGE_HEIGHT / stride[1])},
            {3, int(IMAGE_WIDTH / stride[2]), int(IMAGE_HEIGHT / stride[2])},
    };
    refer_rows = 0;
    refer_cols = 6;
    for (const std::vector<int> &grid : grids) {
        refer_rows += std::accumulate(grid.begin(), grid.end(), 1, std::multiplies<int>());
    }
    GenerateReferMatrix();
    class_colors.resize(CATEGORY);
    srand((int) time(nullptr));
    for (cv::Scalar &class_color : class_colors)
        class_color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
}

YOLOv4::~YOLOv4() = default;

void YOLOv4::LoadEngine() {
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

bool YOLOv4::InferenceFolder(const std::string &folder_name) {
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

    int outSize = bufferSize[1] / sizeof(float) / BATCH_SIZE;

    EngineInference(sample_images, outSize, buffers, bufferSize, stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    // destroy the engine
    context->destroy();
    engine->destroy();
}

void YOLOv4::EngineInference(const std::vector<std::string> &image_list, const int &outSize, void **buffers,
                             const std::vector<int64_t> &bufferSize, cudaStream_t stream) {
    int index = 0;
    int batch_id = 0;
    std::vector<cv::Mat> vec_Mat(BATCH_SIZE);
    std::vector<std::string> vec_name(BATCH_SIZE);
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
//            float out[outSize * BATCH_SIZE * 2];
            float *out = new float[outSize * BATCH_SIZE];
            cudaMemcpyAsync(out, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            auto boxes = postProcess(vec_Mat, out, outSize);
            auto r_end = std::chrono::high_resolution_clock::now();
            float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
            std::cout << "Post process take: " << total_res << " ms." << std::endl;
            total_time += total_res;
            for (int i = 0; i < (int)vec_Mat.size(); i++)
            {
                auto org_img = vec_Mat[i];
                auto rects = boxes[i];
                cv::cvtColor(org_img, org_img, cv::COLOR_BGR2RGB);
                for(const auto &rect : rects)
                {
                    char t[256];
                    sprintf(t, "%.2f", rect.prob);
                    std::string name = coco_labels[rect.classes] + "-" + t;
                    cv::putText(org_img, name, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, class_colors[rect.classes], 2);
                    cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
                    cv::rectangle(org_img, rst, class_colors[rect.classes], 2, cv::LINE_8, 0);
                }
                int pos = vec_name[i].find_last_of(".");
                std::string rst_name = vec_name[i].insert(pos, "_");
                std::cout << rst_name << std::endl;
                cv::imwrite(rst_name, org_img);
            }
            vec_Mat = std::vector<cv::Mat>(BATCH_SIZE);
            delete[] out;
        }
    }
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
}

void YOLOv4::GenerateReferMatrix() {
    refer_matrix = cv::Mat(refer_rows, refer_cols, CV_32FC1);
    int position = 0;
    for (int n = 0; n < (int)grids.size(); n++)
    {
        for (int c = 0; c < grids[n][0]; c++)
        {
            std::vector<int> anchor = anchors[n * grids[n][0] + c];
            for (int h = 0; h < grids[n][1]; h++)
                for (int w = 0; w < grids[n][2]; w++)
                {
                    float *row = refer_matrix.ptr<float>(position);
                    row[0] = w;
                    row[1] = grids[n][1];
                    row[2] = h;
                    row[3] = grids[n][2];
                    row[4] = anchor[0];
                    row[5] = anchor[1];
                    position++;
                }
        }
    }
}

std::vector<float> YOLOv4::prepareImage(std::vector<cv::Mat> &vec_img) {
    std::vector<float> result(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNEL);
    float *data = result.data();
    for (const cv::Mat &src_img : vec_img)
    {
        if (!src_img.data)
            continue;
        cv::Mat flt_img;
        cv::resize(src_img, flt_img, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
        flt_img.convertTo(flt_img, CV_32FC3, 1.0 / 255);

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

std::vector<std::vector<YOLOv4::DetectRes>> YOLOv4::postProcess(const std::vector<cv::Mat> &vec_Mat, float *output,
                                                                const int &outSize) {
    std::vector<std::vector<DetectRes>> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat)
    {
        std::vector<DetectRes> result;
        float *out = output + index * outSize;
        cv::Mat result_matrix = cv::Mat(refer_rows, CATEGORY + 5, CV_32FC1, out);
        for (int index = 0; index < refer_rows; index++) {
            DetectRes box;
            float *row = result_matrix.ptr<float>(index);
            auto max_pos = std::max_element(row + 5, row + CATEGORY + 5);
            box.prob = sigmoid(row[4]) * sigmoid(row[max_pos - row]);
            if (box.prob < obj_threshold)
                continue;
            box.classes = max_pos - row - 5;
            float *anchor = refer_matrix.ptr<float>(index);
            box.x = (sigmoid(row[0]) + anchor[0]) / anchor[1] * src_img.cols;
            box.y = (sigmoid(row[1]) + anchor[2]) / anchor[3] * src_img.rows;
            box.w = exp(row[2]) * anchor[4] / IMAGE_WIDTH * src_img.cols;
            box.h = exp(row[3]) * anchor[5] / IMAGE_HEIGHT * src_img.rows;
            result.push_back(box);
        }
        NmsDetect(result);
        vec_result.push_back(result);
        index++;
    }
    return vec_result;
}

void YOLOv4::NmsDetect(std::vector<DetectRes> &detections) {
    sort(detections.begin(), detections.end(), [=](const DetectRes &left, const DetectRes &right) {
        return left.prob > right.prob;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            if (detections[i].classes == detections[j].classes)
            {
                float iou = IOUCalculate(detections[i], detections[j]);
                if (iou > nms_threshold)
                    detections[j].prob = 0;
            }
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const DetectRes &det)
    { return det.prob == 0; }), detections.end());
}

float YOLOv4::IOUCalculate(const YOLOv4::DetectRes &det_a, const YOLOv4::DetectRes &det_b) {
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

float YOLOv4::sigmoid(float in){
    return 1.f / (1.f + exp(-in));
}
