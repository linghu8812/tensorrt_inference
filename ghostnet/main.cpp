#include "ghostnet.h"

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Please design config file and image folder!" << std::endl;
        return -1;
    }
    std::string config_file = argv[1];
    std::string folder_name = argv[2];
    GhostNet GhostNet(config_file);
    GhostNet.LoadEngine();
    GhostNet.InferenceFolder(folder_name);
    return 0;
}