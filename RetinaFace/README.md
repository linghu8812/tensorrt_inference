# RetinaFace Network MXNet Symbol=>ONNX=>TensorRT

## 1.Reference
- **RetinaFace github:** [https://github.com/deepinsight/insightface/tree/master/RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
- **RetinaFace arxiv:** [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)
- **get pretrained model:** [BaiduDrive](https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ) or [Dropbox](https://www.dropbox.com/s/53ftnlarhyrpkg2/retinaface-R50.zip?dl=0)

## 2.Export ONNX Model
```
python3 export_onnx.py
```

## 3.Build RetinaFace_trt Project
```
mkdir build && cd build
cmake ..
make -j
```

## 4.Run RetinaFace_trt
```
./RetinaFace_trt ../config.yaml ../samples
```

## 5.Inference Time Benchmark
resolution|MXNet Symbol|TensorRT|
---|---|---
640x640||13.2ms

## 6.Results
![](prediction.jpg)