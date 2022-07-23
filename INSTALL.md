# **Depends**

## **TensoRT 7.1.3.4**
- [download](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.1/local_repo/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.1.3.4-ga-20200617_1-1_amd64.deb)<br>
- Install
```
os="ubuntu1x04"
tag="cudax.x-trt7.x.x.x-ga-yyyymmdd"
sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-${tag}/7fa2af80.pub

sudo apt-get update
sudo apt-get install tensorrt cuda-nvrtc-x-y
```
## **OpenCV 4.3.0**
- [download](https://github.com/opencv/opencv/archive/4.3.0.zip)
- Install dependence
```
sudo apt-get install cmake git libgtk2.0-dev pkg-config  libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libtbb2  libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
sudo apt-get install qtbase5-dev qtdeclarative5-dev
sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt-get update
sudo apt install libjasper1 libjasper-dev 
```
- make
```
mkdir build && cd build
cmake .. -DWITH_QT=ON -DBUILD_TIFF=ON -DOPENCV_GENERATE_PKGCONFIG=ON -DCMAKE_INSTALL_PREFIX=/usr/local
sudo make -j16
sudo make install
```
- config
```
pkg-config --cflags opencv4
sudo gedit /etc/ld.so.conf.d/opencv.conf
```
and input
```
/usr/local/lib
```
## **yaml-cpp 0.6.3**
- [download](https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.6.3.zip)
- Build static library
```
mkdir build && cd build
cmake ..
make -j
```
- Build shared library
```
mkdir build && cd build
cmake -DYAML_BUILD_SHARED_LIBS=on ..
make -j
```
