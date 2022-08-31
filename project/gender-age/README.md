# Face Gender and Age Network MXNet Symbol=>ONNX=>TensorRT

## 1.Reference
- **github:** [https://github.com/deepinsight/insightface/tree/master/gender-age](https://github.com/deepinsight/insightface/tree/master/gender-age)
- **get pretrained model:** [params](https://github.com/deepinsight/insightface/blob/master/gender-age/model/model-0000.params)

## 2.Export ONNX Model
```
python3 export_onnx.py
```

## 3.Build tensorrt_inference Project
```
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

## 4.Run tensorrt_inference
```
cd ../../bin/
./tensorrt_inference gender-age ../configs/gender-age/config.yaml ../samples/faces_recognition
```
