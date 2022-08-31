# mmpose PyTorch=>ONNX=>TensorRT

## 1.Reference
- **mmpose:** [https://github.com/open-mmlab/mmpose](https://github.com/open-mmlab/mmpose)

## 2.Export ONNX Model
```bash
python3 tools/pytorch2onnx.py configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth --output-file hrnet_w48_coco_256x192.onnx
```

## 3.Build tensorrt_inference Project
```bash
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

## 4.run tensorrt_inference
```bash
cd ../../bin/
./tensorrt_inference mmpose ../configs/mmpose/config.yaml ../samples/pedestrian_keypoints
```

## 5.detect results
![](result.jpg)
