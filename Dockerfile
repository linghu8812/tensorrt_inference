# docker build -t tensorrt_inference:2.0.0 .
FROM nvcr.io/nvidia/tensorrt:23.04-py3

WORKDIR /home/tensorrt_inference/

# apt-get 安装
RUN apt-get update
RUN apt-get install software-properties-common -y
RUN apt-get update && apt-get -y upgrade && apt-get -y install ssh vim build-essential cmake gdb git libgtk2.0-dev pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
    libdc1394-22-dev qtbase5-dev qtdeclarative5-dev python3-pip zip

# opencv 安装
RUN bash -xc "curl -O http://10.58.253.37:6202/opencv-4.7.0.zip && unzip opencv-4.7.0.zip && mv opencv-4.7.0 opencv && \
    curl -O http://10.58.253.37:6202/opencv_contrib-4.7.0.zip && unzip opencv_contrib-4.7.0.zip && mv opencv_contrib-4.7.0 opencv_contrib && \
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
    popd 2>&1 > /dev/null && popd 2>&1 > /dev/null && rm -rf opencv-4.7.0.zip && rm -rf opencv_contrib-4.7.0.zip"

# pip3 安装
RUN python3 -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple && pip3 install torch torchvision mxnet-cu102 onnx-simplifier -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip3 install --ignore-installed -U PyYAML -i https://pypi.tuna.tsinghua.edu.cn/simple
