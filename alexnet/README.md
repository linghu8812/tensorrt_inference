# AlexNet MXNet Gluon=>ONNX=>TensorRT

## 1.Export ONNX Model
```
python3 export_onnx.py
```

## 2.Build alexnet_trt Project
```
mkdir build && cd build
cmake ..
make -j
```

## 4.run alexnet_trt
```
./alexnet_trt ../alexnet.onnx ../alexnet.trt ../samples/ ../label.txt 224 1
```