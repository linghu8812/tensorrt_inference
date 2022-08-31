# EfficientNet Keras=>ONNX=>TensorRT

## 1.Reference
- **efficientnet arxiv:** [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- **efficientnet github:** [https://github.com/qubvel/efficientnet](https://github.com/qubvel/efficientnet)
- **keras2onnx:** [https://github.com/onnx/keras-onnx](https://github.com/onnx/keras-onnx/blob/master/tutorial/TensorFlow_Keras_EfficientNet.ipynb)
- **pypi:** [https://pypi.org/project/efficientnet](https://pypi.org/project/efficientnet)

run this command to install efficientnet
```
pip install efficientnet
```

## 2.Export ONNX Model
```
python3 export_onnx.py
```

## 3.Build efficientnet_trt Project
```
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

## 4.run efficientnet_trt
```
cd ../../bin/
./tensorrt_inference efficientnet ../configs/efficientnet/config.yaml ../samples/classification
```