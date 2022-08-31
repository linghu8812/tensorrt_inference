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
python3 export.py ---weights weights/yolov5s.pt --batch-size 10 --imgsz 640 --include onnx --simplify
```

## 3.Build tensorrt_inference Project
```
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

## 4.Run tensorrt_inference
- inference with yolov5s
```
cd ../../bin/
./tensorrt_inference yolov5 ../configs/yolov5/config.yaml ../samples/detection_segmentation
```
- inference with yolov5s6
```
cd ../../bin/
./tensorrt_inference yolov5 ../configs/yolov5/config_p6.yaml ../samples/detection_segmentation
```

- inference with yolov5-cls
```
cd ../../bin/
./tensorrt_inference yolov5_cls ../configs/yolov5/config_cls.yaml ../samples/classification
```

For more information, please refer this blog: [https://blog.csdn.net/linghu8812/article/details/109322729](https://blog.csdn.net/linghu8812/article/details/109322729)

## 5.Results:
![](prediction.jpg)
