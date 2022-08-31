# yolor PyTorch=>ONNX=>TensorRT

## 1.Reference
- **yolor code:** [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
- **yolor arxiv:** [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)
- get yolor weights from here: [`yolor.pt`](https://github.com/WongKinYiu/yolor/releases/download/v0.1/yolor.pt) [`yolorx.pt`](https://github.com/WongKinYiu/yolor/releases/download/v0.1/yolorx.pt) [`yolor-w6.pt`](https://github.com/WongKinYiu/yolor/releases/download/v0.1/yolor-w6.pt) [`yolor-e6.pt`](https://github.com/WongKinYiu/yolor/releases/download/v0.1/yolor-e6.pt) [`yolor-d6.pt`](https://github.com/WongKinYiu/yolor/releases/download/v0.1/yolor-d6.pt) [`yolor-e6e.pt`](https://github.com/WongKinYiu/yolor/releases/download/v0.1/yolor-e6e.pt)

For more information, please refer this blog: [https://blog.csdn.net/linghu8812/article/details/125741951?spm=1001.2014.3001.5501](https://blog.csdn.net/linghu8812/article/details/125741951?spm=1001.2014.3001.5501)

## 2.Export ONNX Model
Use the following command to export onnx model:
first download yolor models to folder `weights`,
```bash
git clone https://github.com/linghu8812/yolor.git
cd yolor
python export.py --weights ./weights/yolor.pt
```
if you want to export onnx model with 1280 image size add `--img-size` in command:
```bash
python export.py --weights ./weights/yolor-w6.pt --simplify --grid --img-size 1280
```

## 3.Build yolor_trt Project
```bash
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

## 4.Run yolor_trt
- inference with yolor
```bash
cd ../../bin/
./tensorrt_inference yolor ../configs/yolor/config.yaml ../samples/detection_segmentation
```

## 5.Results:
![](prediction.jpg)
