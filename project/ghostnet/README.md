# ghostnet PyTorch=>ONNX=>TensorRT

## 1.Reference
- **arxiv:** [GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907)
- **github:** [https://github.com/huawei-noah/ghostnet](https://github.com/huawei-noah/ghostnet)
- get ghostnet weights from here: [ghostnet/pytorch](https://github.com/huawei-noah/ghostnet/blob/master/pytorch/models/state_dict_93.98.pth)

## 2.Export ONNX Model
```
python3 export_onnx.py
```

## 3.Build ghostnet_trt Project
```
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

## 4.run ghostnet_trt
```
cd ../../bin/
./tensorrt_inference ghostnet ../configs/ghostnet/config.yaml ../samples/classification
```