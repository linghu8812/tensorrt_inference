# YOLOv4 Network darknet=>ONNX=>TensorRT

## 1.Reference
- **YOLOv4:** [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)
- **darknet:** [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
- get YOLOv4 weights from here: [yolov4.weights](https://drive.google.com/open?id=1sWNozS0emz7bmQTUWDLvsubLGnCwUiIS)

## 2.Export ONNX Model
- export yolov4 onnx
```
python3 export_onnx.py
```
- export yolov4-tiny onnx
```
python3 export_onnx.py --cfg_file yolov4-tiny.cfg --weights_file yolov4-tiny.weights --output_file yolov4-tiny.onnx --strides 32 16 --neck FPN
```
- export yolov3, yolov3-spp onnx
```
python3 export_onnx.py --cfg_file yolov3.cfg --weights_file yolov3.weights --output_file yolov3.onnx --strides 32 16 8 --neck FPN
python3 export_onnx.py --cfg_file yolov3-spp.cfg --weights_file yolov3-spp.weights --output_file yolov3-spp.onnx --strides 32 16 8 --neck FPN
```
- export yolov3-tiny onnx
```
python3 export_onnx.py --cfg_file yolov3-tiny.cfg --weights_file yolov3-tiny.weights --output_file yolov3-tiny.onnx --strides 32 16 --neck FPN
```

## 3.Build Yolov4_trt Project
```
mkdir build && cd build
cmake ..
make -j
```

## 4.Run Yolov4_trt
- inference with yolov4
```
./Yolov4_trt ../config.yaml ../samples
```
- inference with yolov4-tiny
```
./Yolov4_trt ../config-tiny.yaml ../samples
```

## 5.Results
![](prediction.jpg)