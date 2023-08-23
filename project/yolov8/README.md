# YOLOv8 ultralytics=>ONNX=>TensorRT

## 1.Reference
- **YOLOv8:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- docs: [https://docs.ultralytics.com](https://docs.ultralytics.com)
- get YOLOv8 weights from here: [Models](https://github.com/ultralytics/assets/releases).

## 2.Export ONNX Model

- CLI

```bash
yolo export model=yolov8n.pt format=onnx  # export official model
yolo export model=path/to/best.pt format=onnx  # export custom trained model
```

- Python
```bash
python export_onnx.py
```

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom trained

# Export the model
model.export(format="onnx")
```

## 3.Build tensorrt_inference Project
```
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

## 4.Run tensorrt_inference
- inference with yolov8s
```
cd ../../bin/
./tensorrt_inference yolov8 ../configs/yolov8/config.yaml ../samples/detection_segmentation
```
- inference with yolov8x6
```
cd ../../bin/
./tensorrt_inference yolov8 ../configs/yolov8/config_p6.yaml ../samples/detection_segmentation
```

## 5.Results:
![](prediction.jpg)
