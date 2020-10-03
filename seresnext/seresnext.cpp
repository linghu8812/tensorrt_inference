#include <iostream>
#include <fstream>
#include <dirent.h>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "common.hpp"

std::map<int, std::string> imagenet_label;

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
    float f_mean[3] = { 0.485, 0.456, 0.406 };
    float f_std[3]  = { 0.229, 0.224, 0.225 };
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
            split_img[i] = (split_img[i] - f_mean[i]) / f_std[i];
            memcpy(data, split_img[i].data, channelLength * sizeof(float));
            data += channelLength;
        }
    }
    return result;
}

void EngineInference(const std::vector<std::string> &image_list, const int &outSize, const int &BATCH_SIZE, const int &DETECT_WIDTH,
        const int &DETECT_HEIGHT, const int &INPUT_CHANNEL, void *buffers[], const std::vector<int64_t> &bufferSize, cudaStream_t stream,
        nvinfer1::IExecutionContext *context, std::map<int, std::string> imagenet_labels)
{
    int index = 0;
    int batch_id = 0;
    std::vector<cv::Mat> vec_Mat(BATCH_SIZE);
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
            float out[outSize * BATCH_SIZE];
            cudaMemcpyAsync(out, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            for (int i = 0; i < BATCH_SIZE; i++)
            {
                auto result = std::max_element(out + i * outSize, out + (i + 1) * outSize);
                std::cout << imagenet_labels[result - (out + i * outSize)] << std::endl;
            }

            auto r_end = std::chrono::high_resolution_clock::now();
            float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
            std::cout << "Post process take: " << total_res << " ms." << std::endl;
            total_time += total_res;
            vec_Mat = std::vector<cv::Mat>(BATCH_SIZE);
        }
    }
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
}

void doInferenceFrieza(nvinfer1::ICudaEngine *engine, const std::vector<std::string> &sample_images, const int &BATCH_SIZE,
                       const int &DETECT_WIDTH, const int &DETECT_HEIGHT, const int &INPUT_CHANNEL, std::map<int, std::string> imagenet_labels)
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

    EngineInference(sample_images, outSize, BATCH_SIZE, DETECT_WIDTH, DETECT_HEIGHT, INPUT_CHANNEL,
            buffers, bufferSize, stream, context, imagenet_labels);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    // destroy the engine
    context->destroy();
    engine->destroy();
}

std::map<int, std::string> readImageNetLabel(const std::string &fileName)
{
    std::map<int, std::string> imagenet_label;
    std::ifstream file(fileName);
    if (!file.is_open())
    {
        std::cout << "read file error: " << fileName << std::endl;
    }
    std::string strLine;
    while (getline(file, strLine))
    {
        int pos1 = strLine.find(":");
        std::string first = strLine.substr(0, pos1);
        int pos2 = strLine.find_last_of("'");
        std::string second = strLine.substr(pos1 + 3, pos2 - pos1 - 3);
        imagenet_label.insert({atoi(first.c_str()), second});
    }
    file.close();
    return imagenet_label;
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
    std::map<int, std::string> imagenet_labels = readImageNetLabel(label_file);

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

    doInferenceFrieza(engine, sample_images, batch_size, image_size, image_size, input_channel, imagenet_labels);
    return 0;
}