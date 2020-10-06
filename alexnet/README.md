# AlexNet MXNet Gluon=>ONNX=>TensorRT

## 1.Reference
- **AlexNet:** [Imagenet classification with deep convolutional neural networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

## 2.Export ONNX Model
```
python3 export_onnx.py
```

## 3.Build alexnet_trt Project
```
mkdir build && cd build
cmake ..
make -j
```

## 4.run alexnet_trt
```
./alexnet_trt ../config.yaml ../samples
```