//
// Created by linghu8812 on 2022/8/29.
//

#include "build.h"

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Please design config file and folder name!" << std::endl;
        return -1;
    }
    std::string folder_name = argv[3];
    auto model = build_model(argv);
    model->LoadEngine();
    model->InferenceFolder(folder_name);
    return 0;
}
