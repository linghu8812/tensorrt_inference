# yolov7 PyTorch=>ONNX=>TensorRT

## 1.Reference
- **yolov7 code:** [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- **yolov7 arxiv:** [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)
- get yolov7 weights from here: [`yolov7.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) [`yolov7x.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) [`yolov7-w6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) [`yolov7-e6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) [`yolov7-d6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) [`yolov7-e6e.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt)

For more information, please refer this blog: [https://blog.csdn.net/linghu8812/article/details/125741951?spm=1001.2014.3001.5501](https://blog.csdn.net/linghu8812/article/details/125741951?spm=1001.2014.3001.5501)

## 2.Export ONNX Model
Use the following command to export onnx model:
first download yolov7 models to folder `weights`,
```bash
git clone https://github.com/linghu8812/yolov7.git
cd yolov7
python export.py --weights ./weights/yolov7.pt --simplify --grid 
```
if you want to export onnx model with 1280 image size add `--img-size` in command:
```bash
python export.py --weights ./weights/yolov7-w6.pt --simplify --grid --img-size 1280
```

## 3.Build yolov7_trt Project
```bash
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

## 4.Run yolov7_trt
- inference with yolov7
```bash
cd ../../bin/
./tensorrt_inference yolov7 ../configs/yolov7/config.yaml ../samples/detection_segmentation
```

for model such as yolov7-w6  the config file is like this:
```yaml
yolov7:
    onnx_file:     "../weights/yolov7-w6.onnx"
    engine_file:   "../weights/yolov7-w6.trt"
    labels_file:   "../configs/labels/coco.names"
    BATCH_SIZE:    1
    INPUT_CHANNEL: 3
    IMAGE_WIDTH:   1280
    IMAGE_HEIGHT:  1280
    obj_threshold: 0.4
    nms_threshold: 0.45
    agnostic:      False
    strides:       [8, 16, 32, 64]
    num_anchors:   [3,  3,  3,  3]
```

## 5.Results:
![](prediction.jpg)
