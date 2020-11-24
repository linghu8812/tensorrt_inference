# RetinaFace Network MXNet Symbol=>ONNX=>TensorRT

## 1.Reference
- **RetinaFace arxiv:** [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)
- **RetinaFace github:** [https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace)
- **RetinaFaceAntiCov github:** [https://github.com/deepinsight/insightface/tree/master/detection/RetinaFaceAntiCov](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFaceAntiCov)

## 2. Model Zoo

model|weights
---|---
RetinaFace-R50|[weights](https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ)
RetinaFace-MobileNet0.25|[weights](https://pan.baidu.com/s/1P1ypO7VYUbNAezdvLm2m9w)(nzof)
RetinaFaceAntiCov|[weights](https://pan.baidu.com/s/16ihzPxjTObdbv0D6P6LmEQ)(j3b6)

## 3.Export ONNX Model
- clone RetinaFace code
```
git clone https://github.com/deepinsight/insightface.git
cd insightface
```
copy [export_onnx.py](export_onnx.py) to `./detection/RetinaFace` or `./detection/RetinaFaceAntiCov`
- export resnet50 model
```
python3 export_onnx.py
```
- export mobilenet 0.25 model
```
python3 export_onnx.py  --prefix ./model/mnet.25
```
- export RetinaFaceAntiCov model
```
python3 export_onnx.py  --prefix ./model/mnet_cov2 --network net3l
```

## 4.Build RetinaFace_trt Project
```
mkdir build && cd build
cmake ..
make -j
```

## 5.Run RetinaFace_trt
- inference with RetinaFace
```
./RetinaFace_trt ../config.yaml ../samples
```
- inference with RetinaFaceAntiCov
```
./RetinaFace_trt ../config_anti.yaml ../samples
```

## 6.Results
- RetinaFace result

![](prediction_R50.jpg)

- RetinaFaceAntiCov result

![](prediction_Anti.jpg)
