# Face Key Points Network MXNet Symbol=>ONNX=>TensorRT

## 1.Reference
- **github:** [https://github.com/deepinsight/insightface/tree/master/alignment/coordinateReg](https://github.com/deepinsight/insightface/tree/master/alignment/coordinateReg)
- **get pretrained model:** [BaiduDrive](https://pan.baidu.com/s/10m5GmtNV5snynDrq3KqIdg)(code: lqvv) or [GoogleDrive](https://drive.google.com/file/d/1MBWbTEYRhZFzj_O2f2Dc6fWGXFWtbMFw/view?usp=sharing)

## 2.Export ONNX Model
```
python3 export_onnx.py
```

## 3.Build face_alignment_trt Project
```
mkdir build && cd build
cmake ..
make -j
```

## 4.Run face_alignment_trt
```
./face_alignment_trt ../../../configs/face_alignment/config.yaml ../../../samples/faces_recognition
```

## 5.Results
![](prediction.jpg)
