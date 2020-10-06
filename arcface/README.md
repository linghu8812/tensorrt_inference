# ArcFace MXNet Symbol=>ONNX=>TensorRT

## 1.Reference
- **arcface github:** [https://github.com/deepinsight/insightface/tree/master/recognition](https://github.com/deepinsight/insightface/tree/master/recognition)
- **arcface arxiv:** [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- **get pretrained model:** [BaiduDrive](https://pan.baidu.com/s/1wuRTf2YIsKt76TxFufsRNA) and [Dropbox](https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=0)


## 2.Export ONNX Model
```
python3 export_onnx.py --input_shape 4 3 112 112
```

## 3.Build arcface_trt Project
```
mkdir build && cd build
cmake ..
make -j
```

## 4.run arcface_trt
```
./arcface_trt ../config.yaml ../samples
```
**output:**
```
The similarity matrix of the image folder is:
[1, 0.83112532, 0.44810191, 0.44390032;
 0.83112532, 1, 0.43372691, 0.43485713;
 0.44810191, 0.43372691, 1, 0.81798339;
 0.44390032, 0.43485713, 0.81798339, 1]!
```