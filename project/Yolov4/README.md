# YOLOv4 Network darknet=>ONNX=>TensorRT

## 1.Reference
- **YOLOv4:** [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)
- **darknet:** [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)

## 2. darknet model zoo
model|weights
---|---
yolov4|[weights](https://drive.google.com/open?id=1sWNozS0emz7bmQTUWDLvsubLGnCwUiIS)
yolov4-tiny|[weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)
yolov4x-mish|[weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4x-mish.weights)
yolov4-csp|[weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.weights)
yolov3|[weights](https://pjreddie.com/media/files/yolov3.weights)
yolov3-spp|[weights](https://pjreddie.com/media/files/yolov3-spp.weights)
yolov3-tiny|[weights](https://pjreddie.com/media/files/yolov3-tiny.weights)

Above in the table are  now supported models, for more darknet models, please see the official [model zoo](https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo).

## 3.Export ONNX Model
- export yolov4 onnx
```
python3 export_onnx.py
```
- export yolov4-tiny onnx
```
python3 export_onnx.py --cfg_file cfg/yolov4-tiny.cfg --weights_file yolov4-tiny.weights --output_file yolov4-tiny.onnx --strides 32 16 --neck FPN
```
- export yolov4x-mish, yolov4-csp onnx
```
python3 export_onnx.py --cfg_file cfg/yolov4x-mish.cfg --weights_file yolov4x-mish.weights --output_file yolov4x-mish.onnx
python3 export_onnx.py --cfg_file cfg/yolov4-csp.cfg --weights_file yolov4-csp.weights --output_file yolov4-csp.onnx
```
- export yolov3, yolov3-spp onnx
```
python3 export_onnx.py --cfg_file cfg/yolov3.cfg --weights_file yolov3.weights --output_file yolov3.onnx --strides 32 16 8 --neck FPN
python3 export_onnx.py --cfg_file cfg/yolov3-spp.cfg --weights_file yolov3-spp.weights --output_file yolov3-spp.onnx --strides 32 16 8 --neck FPN
```
- export yolov3-tiny onnx
```
python3 export_onnx.py --cfg_file cfg/yolov3-tiny.cfg --weights_file yolov3-tiny.weights --output_file yolov3-tiny.onnx --strides 32 16 --neck FPN
```

## 4.Build Yolov4_trt Project
```
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

## 5.Run Yolov4_trt
- inference with yolov4
```
cd ../../bin/
./tensorrt_inference Yolov4 ../configs/Yolov4/config.yaml ../samples/detection_segmentation
```
- inference with yolov4-tiny
```
cd ../../bin/
./tensorrt_inference Yolov4 ../configs/Yolov4/config-tiny.yaml ../samples/detection_segmentation
```
- inference with yolov4x-mish
```
cd ../../bin/
./tensorrt_inference Yolov4 ../configs/Yolov4/config-xmish.yaml ../samples/detection_segmentation
```

## 6.Results
![](prediction.jpg)

For more information, please see this blog: [https://blog.csdn.net/linghu8812/article/details/109270320](https://blog.csdn.net/linghu8812/article/details/109270320)
