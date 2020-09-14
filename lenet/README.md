# LeNet Network PyTorch=>ONNX=>TensorRT

## 1.Train PyTorch Network
```
python3 train.py
```

## 2.Export ONNX Model
```
python3 export_onnx.py
```

## 3.Build lenet_trt Project
```
mkdir build && cd build
cmake ..
make -j
```

## 4.run lenet_trt
```
./lenet_trt ../mnist_net.onnx ../mnist_net.trt ../samples/ 28 10
```