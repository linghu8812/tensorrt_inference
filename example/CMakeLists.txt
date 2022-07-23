cmake_minimum_required(VERSION 3.5)

project(example)

set(CMAKE_CXX_STANDARD 14)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")

# TensorRT
set(TENSORRT_ROOT /usr/src/tensorrt/)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "Find OpenCV include at ${OpenCV_INCLUDE_DIRS}")

set(MODEL_INCLUDE ../code/includes)
set(MODEL_LIB_DIR libs)
set(YAML_INCLUDE ../depends/yaml-cpp/include)
set(YAML_LIB_DIR ../depends/yaml-cpp/libs)

include_directories(${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${MODEL_INCLUDE} ${YAML_INCLUDE})
link_directories(${MODEL_LIB_DIR} ${YAML_LIB_DIR})

add_executable(example main.cpp)
target_link_libraries(example yaml-cpp yolov5)