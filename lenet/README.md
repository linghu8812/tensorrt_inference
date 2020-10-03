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
mkdir build && cd build
cmake ..
make -j
```

## 5.run lenet_trt
```
./lenet_trt ../mnist_net.onnx ../mnist_net.trt ../samples/ 28 10
```