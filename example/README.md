# **TensorRT Inference Example**

After build `tensorrt_inference` and shared libraries, if you want to use **yolov5** model, you can copy yolov5 library into `libs` folder and build this example project.

```
cd example
mkdir libs build
cp ../bin/libyolov5.so ./libs/
cd build/
cmake ..
make -j
./example ../config.yaml ../../samples/detection_segmentation
```
