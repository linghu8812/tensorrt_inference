# SEResNeXt101 PyTorch=>ONNX=>TensorRT

## 1.Export ONNX Model
```
python3 export_onnx.py
```

## 2.Build seresnext_trt Project
```
mkdir build && cd build
cmake ..
make -j
```

## 4.run seresnext_trt
```
./seresnext_trt ../se_resnext101_32x4d.onnx ../se_resnext101_32x4d.trt ../samples/ ../label.txt 224 1
```