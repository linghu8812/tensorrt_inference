# AlexNet MXNet Gluon=>ONNX=>TensorRT

## 1.Reference
- **AlexNet:** [Imagenet classification with deep convolutional neural networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

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

## 4.run tensorrt_inference
```
cd ../../bin/
./tensorrt_inference alexnet ../configs/alexnet/config.yaml ../samples/classification
```