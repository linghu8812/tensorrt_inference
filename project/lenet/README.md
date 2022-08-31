# LeNet Network PyTorch=>ONNX=>TensorRT

## 1.Reference
- **LeNet:** [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

## 2.Train PyTorch Network
```
python3 train.py
```

## 3.Export ONNX Model
```
python3 export_onnx.py
```

## 4.Build lenet_trt Project
```
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

## 5.Run lenet_trt
```
cd ../../bin/
./tensorrt_inference lenet ../configs/lenet/config.yaml ../samples/mnist
```

## 6.Benchmark(2080Ti)
model|PyTorch|TensorRT|
---|---|---
inference time|0.4ms|0.046ms
