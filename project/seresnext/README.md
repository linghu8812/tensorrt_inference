# SEResNeXt101 PyTorch=>ONNX=>TensorRT

## 1.Reference
- **ResNeXt:** [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
- **SENet:** [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- **Pretrained models:** [https://data.lip6.fr/cadene/pretrainedmodels/](https://data.lip6.fr/cadene/pretrainedmodels/)

## 2.Export ONNX Model
```
python3 export_onnx.py
```

## 3.Build seresnext_trt Project
```
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

## 4.run seresnext_trt
```
cd ../../bin/
./tensorrt_inference seresnext ../configs/seresnext/config.yaml ../samples/classification
```