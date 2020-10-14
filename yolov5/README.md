# YOLOv5 PyTorch=>ONNX=>TensorRT

## 1.Reference
- **yolov5:** [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- get yolov5 weights from here: [yolov5s](https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5s.pt), 
[yolov5m](https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5m.pt), 
[yolov5l](https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5l.pt), 
[yolov5x](https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5x.pt).

## 2.Export ONNX Model
```
git clone https://github.com/linghu8812/yolov5.git
```
copy [export_onnx.py](export_onnx.py) into `yolov5/models` and run `export_onnx.py` to generate `yolov5s.onnx` and so on.
```
export PYTHONPATH="$PWD" && python3 export_onnx.py --weights ./weights/yolov5s.pt --img 640 --batch 1
```

## 3.Build yolov5_trt Project
```
mkdir build && cd build
cmake ..
make -j
```

## 4.run yolov5_trt
```
./yolov5_trt ../config.yaml ../samples
```

## 5.Inference Time Benchmark
model|resolution|PyTorch|TensorRT|
---|---|---|---
yolov5s|640x640| |2.5ms
yolov5x|640x640| |11.3ms