# CenterFace Network ONNX=>TensorRT

## 1.Reference
- **CenterFace github:** [https://github.com/Star-Clouds/CenterFace](https://github.com/Star-Clouds/CenterFace)
- **CenterFace arxiv:** [CenterFace: Joint Face Detection and Alignment Using Face as Point](https://arxiv.org/abs/1911.03599)
- **get pretrained model:** [centerface.onnx](https://github.com/Star-Clouds/CenterFace/blob/master/models/onnx/centerface.onnx) or [centerface_bnmerged.onnx](https://github.com/Star-Clouds/CenterFace/blob/master/models/onnx/centerface_bnmerged.onnx)

## 2.Export ONNX Model
- export `centerface.onnx `model
```
python3 export_onnx.py
```
- export `centerface_bnmerged.onnx `model
```
python3 export_onnx.py --pretrained ./centerface_bnmerged.onnx --onnx ./centerface_bnmerged_transpose.onnx
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
./tensorrt_inference CenterFace ../configs/CenterFace/config.yaml ../samples/faces_detection
```

## 5.Results
![](prediction.jpg)