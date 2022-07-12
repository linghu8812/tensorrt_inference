#include "yolov7.h"

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Please design config file and folder name!" << std::endl;
        return -1;
    }
    std::string config_file = argv[1];
    std::string folder_name = argv[2];
    yolov7 yolov7(config_file);
    yolov7.LoadEngine();
    yolov7.InferenceFolder(folder_name);
    return 0;
}