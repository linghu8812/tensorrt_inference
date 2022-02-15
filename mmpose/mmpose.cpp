#include "mmpose.h"
#include "yaml-cpp/yaml.h"
#include "common.h"

mmpose::mmpose(const std::string &config_file) {
    YAML::Node root = YAML::LoadFile(config_file);
    YAML::Node config = root["mmpose"];
    onnx_file = config["onnx_file"].as<std::string>();
    engine_file = config["engine_file"].as<std::string>();
    BATCH_SIZE = config["BATCH_SIZE"].as<int>();
    INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
    IMAGE_WIDTH = config["IMAGE_WIDTH"].as<int>();
    IMAGE_HEIGHT = config["IMAGE_HEIGHT"].as<int>();
    img_mean = config["img_mean"].as<std::vector<float>>();
    img_std = config["img_mean"].as<std::vector<float>>();
    num_key_points = config["num_key_points"].as<int>();
    skeleton = config["skeleton"].as<std::vector<std::vector<int>>>();
    point_thresh = config["point_thresh"].as<float>();
}

mmpose::~mmpose() = default;

bool mmpose::InferenceFolder(const std::string &folder_name) {
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

void mmpose::EngineInference(const std::vector<std::string> &image_list, const int &outSize, void **buffers,
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
            cudaStreamSynchronize(stream);
            
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
            std::vector<std::vector<KeyPoint>> key_points = postProcess(vec_Mat, out, outSize);
            auto r_end = std::chrono::high_resolution_clock::now();
            float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
            std::cout << "Post process take: " << total_res << " ms." << std::endl;
            total_time += total_res;
            for (int i = 0; i < (int)vec_Mat.size(); i++)
            {
                auto org_img = vec_Mat[i];
                if (!org_img.data)
                    continue;
                auto current_points = key_points[i];
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
                int pos = vec_name[i].find_last_of(".");
                std::string rst_name = vec_name[i].insert(pos, "_");
                std::cout << rst_name << std::endl;
                cv::imwrite(rst_name, org_img);
            }
            vec_Mat = std::vector<cv::Mat>(BATCH_SIZE);
        }
    }
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
}

std::vector<float> mmpose::prepareImage(std::vector<cv::Mat> &vec_img) {
    std::vector<float> result(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNEL);
    float *data = result.data();
    for (const cv::Mat &src_img : vec_img)
    {
        if (!src_img.data)
            continue;
        float ratio = std::min(float(IMAGE_WIDTH) / float(src_img.cols), float(IMAGE_HEIGHT) / float(src_img.rows));
        cv::Mat flt_img = cv::Mat::zeros(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3);
        cv::Mat rsz_img;
        cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);
        rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
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

std::vector<std::vector<mmpose::KeyPoint>> mmpose::postProcess(const std::vector<cv::Mat> &vec_Mat, float *output, const int &outSize) {
    std::vector<std::vector<KeyPoint>> vec_key_points;
    int feature_size = IMAGE_WIDTH * IMAGE_HEIGHT / 16;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat) {
        std::vector<KeyPoint> key_points = std::vector<KeyPoint>(num_key_points);
        float ratio = std::max(float(src_img.cols) / float(IMAGE_WIDTH), float(src_img.rows) / float(IMAGE_HEIGHT));
        float *current_person = output + index * outSize;
        for (int number = 0; number < num_key_points; number++) {
            float *current_point = current_person + feature_size * number;
            auto max_pos = std::max_element(current_point, current_point + feature_size);
            key_points[number].prob = *max_pos;
            float x = (max_pos - current_point) % (IMAGE_WIDTH / 4) + (*(max_pos + 1) > *(max_pos - 1) ? 0.25 : -0.25);
            float y = (max_pos - current_point) / (IMAGE_WIDTH / 4) + (*(max_pos + IMAGE_WIDTH / 4) > *(max_pos - IMAGE_WIDTH / 4) ? 0.25 : -0.25);
            key_points[number].x = int(x * ratio * 4);
            key_points[number].y = int(y * ratio * 4);
            key_points[number].number = number;
        }
        vec_key_points.push_back(key_points);
        index++;
    }


    return vec_key_points;
}
