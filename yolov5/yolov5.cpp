#include <iostream>
#include <fstream>
#include <dirent.h>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "common.hpp"

struct DetectionRes{
    int classes;
    float x;
    float y;
    float w;
    float h;
    float prob;
};

std::vector<std::string>readFolder(const std::string &image_path)
{
    std::vector<std::string> image_names;
    auto dir = opendir(image_path.c_str());

    if ((dir) != nullptr)
    {
        struct dirent *entry;
        entry = readdir(dir);
        while (entry)
        {
            auto temp = image_path + "/" + entry->d_name;
            if (strcmp(entry->d_name, "") == 0 || strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            {
                entry = readdir(dir);
                continue;
            }
            image_names.push_back(temp);
            entry = readdir(dir);
        }
    }
    return image_names;
}

bool readTrtFile(const std::string &engineFile, //name of the engine file
                 nvinfer1::ICudaEngine *&engine)
{
    std::string cached_engine;
    std::fstream file;
    std::cout << "loading filename from:" << engineFile << std::endl;
    nvinfer1::IRuntime *trtRuntime;
    file.open(engineFile, std::ios::binary | std::ios::in);

    if (!file.is_open()) {
        std::cout << "read file error: " << engineFile << std::endl;
        cached_engine = "";
    }

    while (file.peek() != EOF) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        cached_engine.append(buffer.str());
    }
    file.close();

    trtRuntime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());
    engine = trtRuntime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
    std::cout << "deserialize done" << std::endl;

    return true;
}

void onnxToTRTModel(const std::string &modelFile, // name of the onnx model
                    const std::string &filename,  // name of saved engine
                    nvinfer1::ICudaEngine *&engine, const int &BATCH_SIZE)
{
    // create the builder
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    auto config = builder->createBuilderConfig();

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    if (!parser->parseFromFile(modelFile.c_str(), static_cast<int>(gLogger.getReportableSeverity()))) {
        gLogError << "Failure while parsing ONNX file" << std::endl;
    }
    // Build the engine
    builder->setMaxBatchSize(BATCH_SIZE);
    config->setMaxWorkspaceSize(16_MiB);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    std::cout << "start building engine" << std::endl;
    engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build engine done" << std::endl;
    assert(engine);
    // we can destroy the parser
    parser->destroy();
    // save engine
    nvinfer1::IHostMemory *data = engine->serialize();
    std::ofstream file;
    file.open(filename, std::ios::binary | std::ios::out);
    std::cout << "writing engine file..." << std::endl;
    file.write((const char *) data->data(), data->size());
    std::cout << "save engine file done" << std::endl;
    file.close();
    // then close everything down
    network->destroy();
    builder->destroy();
}

std::vector<float> prepareImage(const std::vector<cv::Mat> &vec_img, const int &BATCH_SIZE, const int &DETECT_WIDTH, const int &DETECT_HEIGHT, const int &INPUT_CHANNEL)
{
    std::vector<float> result(BATCH_SIZE * DETECT_HEIGHT * DETECT_WIDTH * INPUT_CHANNEL);
    float *data = result.data();
    for (const cv::Mat &src_img : vec_img)
    {
        if (!src_img.data)
            continue;
        cv::Mat flt_img;
        cv::resize(src_img, flt_img, cv::Size(DETECT_HEIGHT, DETECT_WIDTH));
        flt_img.convertTo(flt_img, CV_32FC3, 1.0 / 255);

        //HWC TO CHW
        std::vector<cv::Mat> split_img(INPUT_CHANNEL);
        cv::split(flt_img, split_img);

        int channelLength = DETECT_HEIGHT * DETECT_WIDTH;
        for (int i = 0; i < INPUT_CHANNEL; ++i)
        {
            memcpy(data, split_img[i].data, channelLength * sizeof(float));
            data += channelLength;
        }
    }
    return result;
}

float iou_calculate(const DetectionRes &det_a, const DetectionRes &det_b)
{
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

void DoNms(std::vector <DetectionRes> &detections, float nms_threshold) {

    sort(detections.begin(), detections.end(), [=](const DetectionRes &left, const DetectionRes &right) {
        return left.prob > right.prob;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            if (detections[i].classes == detections[j].classes)
            {
                float iou = iou_calculate(detections[i], detections[j]);
                if (iou > nms_threshold)
                    detections[j].prob = 0;
            }
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const DetectionRes &det)
    { return det.prob == 0; }), detections.end());
}

std::vector<std::vector<DetectionRes>> postProcess(const std::vector<cv::Mat> &vec_Mat, float *output, const int &outSize,
        const int &CATEGORY, const std::vector<std::vector<int>> &grids, const std::vector<std::vector<int>> &anchors,
        const int &DETECT_WIDTH, const int &DETECT_HEIGHT, const float &obj_threshold, const float &nms_threshold)
{
    std::vector<std::vector<DetectionRes>> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat)
    {
        std::vector<DetectionRes> result;
        float *out = output + index * outSize;
        int position = 0;
        for (int n = 0; n < (int)grids.size(); n++)
        {
            for (int c = 0; c < grids[n][0]; c++)
            {
                std::vector<int> anchor = anchors[n * grids[n][0] + c];
                for (int h = 0; h < grids[n][1]; h++)
                    for (int w = 0; w < grids[n][2]; w++)
                    {
                        float *row = out + position * (CATEGORY + 5);
                        position++;
                        DetectionRes box;
                        auto max_pos = std::max_element(row + 5, row + CATEGORY + 5);
                        box.prob = row[4] * row[max_pos - row];
                        if (box.prob < obj_threshold)
                            continue;
                        box.classes = max_pos - row - 5;
                        box.x = (row[0] * 2 - 0.5 + w) / grids[n][1] * src_img.cols;
                        box.y = (row[1] * 2 - 0.5 + h) / grids[n][2] * src_img.rows;
                        box.w = pow(row[2] * 2, 2) * anchor[0] / DETECT_WIDTH * src_img.cols;
                        box.h = pow(row[3] * 2, 2) * anchor[1] / DETECT_HEIGHT * src_img.rows;
                        result.push_back(box);
                    }
            }
        }
        DoNms(result, nms_threshold);
        vec_result.push_back(result);
        index++;
    }
    return vec_result;
}

void EngineInference(const std::vector<std::string> &image_list, const int &outSize, const int &BATCH_SIZE, const int &DETECT_WIDTH,
        const int &DETECT_HEIGHT, const int &INPUT_CHANNEL, const int &CATEGORY, void *buffers[], const std::vector<int64_t> &bufferSize, cudaStream_t stream,
        nvinfer1::IExecutionContext *context, std::map<int, std::string> coco_labels)
{
    float obj_threshold = 0.4;
    float nms_threshold = 0.45;
    std::vector<int> stride = { 8, 16, 32 };
    std::vector<std::vector<int>> anchors = {
            {10, 13}, {16, 30}, {33, 23}, {30, 61}, {62, 45}, {59, 119}, {116, 90}, {156, 198}, {373, 326}
    };
    std::vector<std::vector<int>> grids = {
            {3, int(DETECT_WIDTH / stride[0]), int(DETECT_HEIGHT / stride[0])},
            {3, int(DETECT_WIDTH / stride[1]), int(DETECT_HEIGHT / stride[1])},
            {3, int(DETECT_WIDTH / stride[2]), int(DETECT_HEIGHT / stride[2])},
    };

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
            std::vector<float>curInput = prepareImage(vec_Mat, BATCH_SIZE, DETECT_WIDTH, DETECT_HEIGHT, INPUT_CHANNEL);
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

            auto boxes = postProcess(vec_Mat, out, outSize, CATEGORY, grids, anchors, DETECT_WIDTH, DETECT_HEIGHT,
                                     obj_threshold, nms_threshold);


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
                    std::cout << name << std::endl;
                    cv::putText(org_img, name, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
                    cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
                    cv::rectangle(org_img, rst, cv::Scalar(255, 0, 0), 2, cv::LINE_8, 0);
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

void doInferenceFrieza(nvinfer1::ICudaEngine *engine, const std::vector<std::string> &sample_images, const int &BATCH_SIZE,
                       const int &DETECT_WIDTH, const int &DETECT_HEIGHT, const int &INPUT_CHANNEL, const int &CATEGORY,
                       std::map<int, std::string> imagenet_labels)
{
    //get context
    assert(engine != nullptr);
    nvinfer1::IExecutionContext *context = engine->createExecutionContext();
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

    EngineInference(sample_images, outSize, BATCH_SIZE, DETECT_WIDTH, DETECT_HEIGHT, INPUT_CHANNEL, CATEGORY,
            buffers, bufferSize, stream, context, imagenet_labels);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    // destroy the engine
    context->destroy();
    engine->destroy();
}

std::map<int, std::string> readCOCOLabel(const std::string &fileName)
{
    std::map<int, std::string> coco_label;
    std::ifstream file(fileName);
    if (!file.is_open())
    {
        std::cout << "read file error: " << fileName << std::endl;
    }
    std::string strLine;
    int index = 0;
    while (getline(file, strLine))
    {
        coco_label.insert({index, strLine});
        index++;
    }
    file.close();
    return coco_label;
}

int main(int argc, char **argv)
{
    //Parse args
    if (argc < 7)
    {
        std::cout << "Please design onnx file, trt file, samples folder, labels file, image size and batch size in order!" << std::endl;
        return -1;
    }

    std::string onnx_file = argv[1];
    std::string engine_file = argv[2];
    std::string sample_folder = argv[3];
    std::string label_file = argv[4];
    int image_size = atoi(argv[5]);
    int batch_size = atoi(argv[6]);

    int input_channel = 3;

    std::vector<std::string> sample_images = readFolder(sample_folder);
    std::map<int, std::string> coco_labels = readCOCOLabel(label_file);

    nvinfer1::ICudaEngine *engine = nullptr;

    // create and load engine
    std::fstream existEngine;
    existEngine.open(engine_file, std::ios::in);
    if (existEngine) {
        readTrtFile(engine_file, engine);
        assert(engine != nullptr);
    } else {
        onnxToTRTModel(onnx_file, engine_file, engine, batch_size);
        assert(engine != nullptr);
    }

    doInferenceFrieza(engine, sample_images, batch_size, image_size, image_size, input_channel, coco_labels.size(),
                      coco_labels);
    return 0;
}