cmake_minimum_required(VERSION 3.5)

project(tensorrt_inference)

set(CMAKE_CXX_STANDARD 14)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

set(MODEL_INCLUDE ../code/includes)
set(YAML_INCLUDE ../depends/yaml-cpp/include)
set(YAML_LIB_DIR ../depends/yaml-cpp/libs)

include_directories(${CUDA_INCLUDE_DIRS} ${MODEL_INCLUDE} ${YAML_INCLUDE})
link_directories(${YAML_LIB_DIR})

add_subdirectory(yolov5)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../bin)
add_executable(tensorrt_inference ../../code/src/build.cpp main.cpp)
target_link_libraries(tensorrt_inference yaml-cpp yolov5)

