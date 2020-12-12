FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

WORKDIR /home/install/

# apt-get 安装
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && apt-get -y upgrade && apt-get -y install ssh vim build-essential cmake git libgtk2.0-dev pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
    libjasper-dev libdc1394-22-dev qtbase5-dev qtdeclarative5-dev python3-pip zip

# opencv 安装
RUN bash -xc "curl -O https://github.com/opencv/opencv/archive/4.3.0.zip && unzip opencv-4.3.0.zip && mv opencv-4.3.0 opencv && \
    curl -O https://github.com/opencv/opencv_contrib/archive/4.3.0.zip && unzip opencv_contrib-4.3.0.zip && mv opencv_contrib-4.3.0 opencv_contrib && \
    pushd opencv>&1 > /dev/null && mkdir build && pushd build>&1 > /dev/null && \
    cmake -D WITH_QT=ON \
          -D WITH_CUDA=ON \
          -D BUILD_TIFF=ON \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ \
          -D BUILD_opencv_xfeatures2d=OFF  .. && \
    make -j4 && make -j4 install && pkg-config --cflags opencv4 && echo '/usr/local/lib' > /etc/ld.so.conf.d/opencv.conf && \
    popd 2>&1 > /dev/null && popd 2>&1 > /dev/null && rm -rf opencv-4.3.0.zip && rm -rf opencv_contrib-4.3.0.zip"

# pip3 安装
RUN pip3 install -U pip && pip3 install torch torchvision mxnet-cu102 onnx-simplifier \
    && pip3 install --ignore-installed -U PyYAML

# tensorrt 安装
RUN bash -xc "curl -O https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.1/local_repo/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.1.3.4-ga-20200617_1-1_amd64.deb && \
    dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.1.3.4-ga-20200617_1-1_amd64.deb \
    && apt-key add /var/nv-tensorrt-repo-cuda10.2-trt7.1.3.4-ga-20200617/7fa2af80.pub && apt-get update\
    && apt-get -y install tensorrt && rm -rf nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.1.3.4-ga-20200617_1-1_amd64.deb"
