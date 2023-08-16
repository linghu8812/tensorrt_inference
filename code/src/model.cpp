//
// Created by linghu8812 on 2021/2/8.
//

#include "model.h"
#include "NvInferPlugin.h"

Model::Model(const YAML::Node &config) {
    onnx_file = config["onnx_file"].as<std::string>();
    engine_file = config["engine_file"].as<std::string>();
    BATCH_SIZE = config["BATCH_SIZE"].as<int>();
    INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
    IMAGE_WIDTH = config["IMAGE_WIDTH"].as<int>();
    IMAGE_HEIGHT = config["IMAGE_HEIGHT"].as<int>();
    image_order = config["image_order"].as<std::string>();
    channel_order = config["channel_order"].as<std::string>();
    img_mean = config["img_mean"].as<std::vector<float>>();
    img_std = config["img_std"].as<std::vector<float>>();
    alpha = config["alpha"].as<float>();
    resize = config["resize"].as<std::string>();
}

Model::~Model() {
    context->destroy();
    engine->destroy();
    builder->destroy();
}

void Model::OnnxToTRTModel() {
    // create the builder
    builder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    auto config = builder->createBuilderConfig();

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(gLogger.getReportableSeverity()))) {
        gLogError << "Failure while parsing ONNX file" << std::endl;
    }
    // Build the engine
    builder->setMaxBatchSize(BATCH_SIZE);
    config->setMaxWorkspaceSize(8_GiB);
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
    file.open(engine_file, std::ios::binary | std::ios::out);
    std::cout << "writing engine file..." << std::endl;
    file.write((const char *) data->data(), data->size());
    std::cout << "save engine file done" << std::endl;
    file.close();
    // then close everything down
    network->destroy();
    config->destroy();
}

bool Model::ReadTrtFile() {
    std::string cached_engine;
    std::fstream file;
    std::cout << "loading filename from:" << engine_file << std::endl;
    nvinfer1::IRuntime *trtRuntime;
    file.open(engine_file, std::ios::binary | std::ios::in);

    if (!file.is_open()) {
        std::cout << "read file error: " << engine_file << std::endl;
        cached_engine = "";
    }

    while (file.peek() != EOF) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        cached_engine.append(buffer.str());
    }
    file.close();

    trtRuntime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());
    initLibNvInferPlugins(&gLogger, "");
    engine = trtRuntime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
    std::cout << "deserialize done" << std::endl;

}

void Model::LoadEngine() {
    // create and load engine
    std::fstream existEngine;
    existEngine.open(engine_file, std::ios::in);
    if (existEngine) {
        ReadTrtFile();
        assert(engine != nullptr);
    } else {
        OnnxToTRTModel();
        assert(engine != nullptr);
    }

    context = engine->createExecutionContext();
    assert(context != nullptr);

    //get buffers
    assert(engine->getNbBindings() == 2);
    int nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);
    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * getElementSize(dtype);
        bufferSize[i] = totalSize;
        std::cout << "binding" << i << ": " << totalSize << std::endl;
        cudaMalloc(&buffers[i], totalSize);
    }
    //get stream
    cudaStreamCreate(&stream);
    outSize = int(bufferSize[1] / sizeof(float) / BATCH_SIZE);
}

std::vector<float> Model::PreProcess(std::vector<cv::Mat> &vec_img) {
    std::vector<float> result(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNEL);
    float *data = result.data();
    for (const cv::Mat &src_img : vec_img) {
        if (!src_img.data)
            continue;
        cv::Mat flt_img;
        if (INPUT_CHANNEL == 1)
            cv::cvtColor(src_img, flt_img, cv::COLOR_RGB2GRAY);
        else if (INPUT_CHANNEL == 3)
            flt_img = src_img.clone();
        if (resize == "directly") {
            cv::resize(flt_img, flt_img, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
        } else if (resize == "keep_ratio") {
            float ratio = std::min(float(IMAGE_WIDTH) / float(src_img.cols), float(IMAGE_HEIGHT) / float(src_img.rows));
            flt_img = cv::Mat::zeros(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3);
            cv::Mat rsz_img;
            cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);
            rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
        }
        if (INPUT_CHANNEL == 1)
            flt_img.convertTo(flt_img, CV_32FC1, 1 / alpha);
        else if (INPUT_CHANNEL == 3)
            flt_img.convertTo(flt_img, CV_32FC3, 1 / alpha);
        std::vector<cv::Mat> split_img(INPUT_CHANNEL);
        cv::split(flt_img, split_img);
        //HWC TO CHW
        if (image_order == "BCHW") {
            int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT;
            for (int i = 0; i < INPUT_CHANNEL; ++i) {
                split_img[i] = (split_img[i] - img_mean[i]) / img_std[i];
                memcpy(data, split_img[i].data, channelLength * sizeof(float));
                data += channelLength;
            }
        } else if (image_order == "BHWC") {
            for (int i = 0; i < INPUT_CHANNEL; ++i)
                split_img[i] = (split_img[i] - img_mean[i]) / img_std[i];
            cv::merge(split_img, flt_img);
            int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNEL;
            memcpy(data, flt_img.data, channelLength * sizeof(float));
        }

    }
    return result;
}

void Model::ModelInference(std::vector<float> image_data, float *output) {
    if (!image_data.data()) {
        std::cout << "prepare images ERROR!" << std::endl;
        return;
    }
    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    cudaMemcpyAsync(buffers[0], image_data.data(), bufferSize[0], cudaMemcpyHostToDevice, stream);

    // do inference
    context->executeV2(buffers);
    cudaMemcpyAsync(output, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}
