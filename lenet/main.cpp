#include "lenet.h"

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Please design video name and config file!" << std::endl;
        return -1;
    }
    std::string config_file = argv[1];
    std::string file_name = argv[2];
    LeNet LeNet(config_file);
    LeNet.LoadEngine();
    LeNet.InferenceFolder(file_name);
    return 0;
}