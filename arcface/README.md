# ArcFace MXNet Symbol=>ONNX=>TensorRT

## 1.Reference
- **arcface github:** [https://github.com/deepinsight/insightface/tree/master/recognition](https://github.com/deepinsight/insightface/tree/master/recognition)
- **arcface arxiv:** [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- **get pretrained model:** [BaiduDrive](https://pan.baidu.com/s/1wuRTf2YIsKt76TxFufsRNA) and [Dropbox](https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=0)


## 2.Export ONNX Model
```
python3 export_onnx.py
```

## 3.Build arcface_trt Project
```
mkdir build && cd build
cmake ..
make -j
```

## 4.run arcface_trt
```
./arcface_trt ../arcface_r100.onnx ../arcface_r100.trt ../samples 112 1
```
**output:**
```
The similarity of the two images is: 0.698439!
```