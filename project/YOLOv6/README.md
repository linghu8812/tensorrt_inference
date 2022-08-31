# YOLOv6 PyTorch=>ONNX=>TensorRT

## 1.Reference
- **YOLOv6:** [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)
- get YOLOv6 weights from here: [YOLOv6-n](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6n.pt), 
[YOLOv6-tiny](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6t.pt), 
[YOLOv6-s](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.pt)

## 2.Export ONNX Model
Use the following command to export onnx model:
```
git clone https://github.com/meituan/YOLOv6.git
cd YOLOv6
python deploy/ONNX/export_onnx.py --weights yolov6s.pt --img 640 --batch 1
```
or download the onnx model directly on the page: [https://github.com/meituan/YOLOv6/releases/tag/0.1.0](https://github.com/meituan/YOLOv6/releases/tag/0.1.0)

## 3.Build YOLOv6_trt Project
```
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

For more information, please refer this blog: [https://blog.csdn.net/linghu8812/article/details/125513929](https://blog.csdn.net/linghu8812/article/details/125513929)

## 4.Run YOLOv6_trt
- inference with yolov6s
```
cd ../../bin/
./tensorrt_inference YOLOv6 ../configs/YOLOv6/config.yaml ../samples/detection_segmentation
```

## 5.Results:
![](prediction.jpg)
