# Fully Convolutional Networks GluonCV=>ONNX=>TensorRT

## 1.Reference
- **FCN:** [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- **GluonCV Tutorial:** [Getting Started with FCN Pre-trained Models](https://cv.gluon.ai/build/examples_segmentation/demo_fcn.html#sphx-glr-build-examples-segmentation-demo-fcn-py)

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
./tensorrt_inference FCN ../configs/FCN/config.yaml ../samples/detection_segmentation
```

## 5.Results
![](origin.jpg)![](prediction.jpg)