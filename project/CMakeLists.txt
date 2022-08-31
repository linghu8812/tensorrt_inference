cmake_minimum_required(VERSION 3.5)

project(tensorrt_inference)

set(CMAKE_CXX_STANDARD 14)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "Find OpenCV include at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Find OpenCV libraries: ${OpenCV_LIBRARIES}")

set(MODEL_INCLUDE ../code/includes)
set(YAML_INCLUDE ../depends/yaml-cpp/include)
set(YAML_LIB_DIR ../depends/yaml-cpp/libs)

include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${MODEL_INCLUDE} ${YAML_INCLUDE})
link_directories(${YAML_LIB_DIR})

add_subdirectory(alexnet)
add_subdirectory(arcface)
add_subdirectory(CenterFace)
add_subdirectory(efficientnet)
add_subdirectory(face_alignment)
add_subdirectory(fast-reid)
add_subdirectory(FCN)
add_subdirectory(gender-age)
add_subdirectory(ghostnet)
add_subdirectory(lenet)
add_subdirectory(MiniFASNet)
add_subdirectory(mmpose)
add_subdirectory(nanodet)
add_subdirectory(RetinaFace)
add_subdirectory(ScaledYOLOv4)
add_subdirectory(scrfd)
add_subdirectory(seresnext)
add_subdirectory(Swin-Transformer)
add_subdirectory(yolor)
add_subdirectory(Yolov4)
add_subdirectory(yolov5)
add_subdirectory(YOLOv6)
add_subdirectory(yolov7)

add_library(factory SHARED ./../code/src/build.cpp)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../bin)
add_executable(tensorrt_inference main.cpp)
target_link_libraries(tensorrt_inference yaml-cpp factory alexnet arcface CenterFace efficientnet face_alignment
        fast-reid FCN gender-age ghostnet lenet MiniFASNet mmpose nanodet RetinaFace ScaledYOLOv4 scrfd seresnext
        Swin-Transformer yolor Yolov4 yolov5 YOLOv6 yolov7)

