#include "FCN.h"
#include "yaml-cpp/yaml.h"
#include "common.hpp"

FCN::FCN(const std::string &config_file) {
    YAML::Node root = YAML::LoadFile(config_file);
    YAML::Node config = root["FCN"];
    onnx_file = config["onnx_file"].as<std::string>();
    engine_file = config["engine_file"].as<std::string>();
    BATCH_SIZE = config["BATCH_SIZE"].as<int>();
    INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
    IMAGE_WIDTH = config["IMAGE_WIDTH"].as<int>();
    IMAGE_HEIGHT = config["IMAGE_HEIGHT"].as<int>();
    OUT_WIDTH = config["OUT_WIDTH"].as<int>();
    OUT_HEIGHT = config["OUT_HEIGHT"].as<int>();
    CATEGORY = config["CATEGORY"].as<int>();
    img_mean = config["img_mean"].as<std::vector<float>>();
    img_std = config["img_std"].as<std::vector<float>>();
    class_colors.resize(CATEGORY);
    srand((int) time(nullptr));
    for (cv::Scalar &class_color : class_colors)
        class_color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
}

FCN::~FCN() = default;

void FCN::LoadEngine() {
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

bool FCN::InferenceFolder(const std::string &folder_name) {
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

void FCN::EngineInference(const std::vector<std::string> &image_list, const int &outSize, void **buffers,
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
            auto *out = new float[outSize * BATCH_SIZE];
            cudaMemcpyAsync(out, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            auto mask = postProcess(vec_Mat, out, outSize);
            delete[] out;
            auto r_end = std::chrono::high_resolution_clock::now();
            float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
            std::cout << "Post process take: " << total_res << " ms." << std::endl;
            total_time += total_res;
            for (int i = 0; i < (int)vec_Mat.size(); i++)
            {
                auto org_img = vec_Mat[i];
                if (!org_img.data)
                    continue;
                auto result = mask[i];
//                cv::cvtColor(org_img, org_img, cv::COLOR_BGR2RGB);
                cv::Mat rsz_mat;
                cv::resize(result, rsz_mat, cv::Size(org_img.cols, org_img.rows));
                std::vector<cv::Mat> vec_rst = {
                        cv::Mat(cv::Size(org_img.cols, org_img.rows), CV_8UC1),
                        cv::Mat(cv::Size(org_img.cols, org_img.rows), CV_8UC1),
                        cv::Mat(cv::Size(org_img.cols, org_img.rows), CV_8UC1)
                };
                for (int h = 0; h < org_img.rows; h++)
                {
                    auto *p = rsz_mat.ptr<uchar>(h);
                    auto *r0 = vec_rst[0].ptr<uchar>(h);
                    auto *r1 = vec_rst[1].ptr<uchar>(h);
                    auto *r2 = vec_rst[2].ptr<uchar>(h);
                    for (int w = 0; w < org_img.cols; w ++)
                    {
                        r0[w] = p[w] == 0 ? 0 : class_colors[p[w] - 1][0];
                        r1[w] = p[w] == 0 ? 0 : class_colors[p[w] - 1][1];
                        r2[w] = p[w] == 0 ? 0 : class_colors[p[w] - 1][2];
                    }
                }
                cv::Mat rst_mat;
                cv::merge(vec_rst, rst_mat);
                int pos = vec_name[i].find_last_of(".");
                std::string rst_name = vec_name[i].insert(pos, "_");
                std::cout << rst_name << std::endl;
                cv::imwrite(rst_name, rst_mat);
            }
            vec_Mat = std::vector<cv::Mat>(BATCH_SIZE);
        }
    }
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
}

std::vector<float> FCN::prepareImage(std::vector<cv::Mat> &vec_img) {
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
            split_img[i] = (split_img[i] - img_mean[i]) / img_std[i];
            memcpy(data, split_img[i].data, channelLength * sizeof(float));
            data += channelLength;
        }
    }
    return result;
}

std::vector<cv::Mat> FCN::postProcess(const std::vector<cv::Mat> &vec_Mat, float *output, const int &outSize) {
    std::vector<cv::Mat> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat)
    {
        cv::Mat result = cv::Mat(OUT_WIDTH, OUT_HEIGHT, CV_8UC1);
        float *out = output + index * outSize;
        for (int h = 0; h < OUT_HEIGHT; h++) {
            uchar *row = result.ptr<uchar>(h);
            for (int w = 0; w < OUT_WIDTH; w++) {
                float *current = out + (h * OUT_WIDTH + w) * (CATEGORY + 1);
                auto max_pos = std::max_element(current, current + CATEGORY + 1);
                row[w] = max_pos - current;
            }
        }
        vec_result.push_back(result);
        index++;
    }
    return vec_result;
}
