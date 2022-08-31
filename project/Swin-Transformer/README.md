# Swin-Transformer timm=>ONNX=>TensorRT

## 1.Reference
- **Swin-Transformer:** [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- **Swin-Transformer github:** [https://github.com/microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- **timm:** [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

## 2.Export ONNX Model
```
python3 export_onnx.py
```

## 3.Build Swin-Transformer_trt Project
```
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

## 4.run Swin-Transformer_trt
```
cd ../../bin/
./tensorrt_inference Swin_Transformer ../configs/Swin-Transformer/config.yaml ../../../samples/classification
```