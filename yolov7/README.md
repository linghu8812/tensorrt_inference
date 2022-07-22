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
mkdir build && cd build
cmake ..
make -j
```

## 4.Run yolov7_trt
- inference with yolov7
```bash
./yolov7_trt ../config.yaml ../samples
```

for model such as yolov7-w6  the config file is like this:
```yaml
yolov7:
    onnx_file:     "../yolov7-w6.onnx"
    engine_file:   "../yolov7-w6.trt"
    labels_file:   "../coco.names"
    BATCH_SIZE:    1
    INPUT_CHANNEL: 3
    IMAGE_WIDTH:   1280
    IMAGE_HEIGHT:  1280
    obj_threshold: 0.4
    nms_threshold: 0.45
    strides:       [8, 16, 32, 64]
    num_anchors:   [3,  3,  3,  3]
```

## 5.Results:
![](prediction.jpg)

## COCO AP evaluation

1. Download COCO validation set images (http://images.cocodataset.org/zips/val2017.zip) and annotations (http://images.cocodataset.org/annotations/annotations_trainval2017.zip), unzip them.

2. Compile and run yolov7 with `EXPORT_COCO_JSON` flag enabled

3. Install the COCO python API, then run `coco_eval.py` (make sure the path in the script are correct)

```
python -m pip install pycocotools
python coco_eval.py
```