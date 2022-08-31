# Nanodet PyTorch=>ONNX=>TensorRT

## 1.Reference
- **nanodet:** [https://github.com/RangiLyu/nanodet](https://github.com/RangiLyu/nanodet)
- get nanodet pretrained weights from here: [COCO pretrain weight for torch>=1.6(Google Drive)](https://drive.google.com/file/d/1EhMqGozKfqEfw8y9ftbi1jhYu86XoW62/view?usp=sharing) | [COCO pretrain weight for torch<=1.5(Google Drive)](https://drive.google.com/file/d/10h-0qLMCgYvWQvKULqbkLvmirFR-w9NN/view?usp=sharing)

## 2.Export ONNX Model
```
git clone https://github.com/linghu8812/nanodet.git
```
copy [export_onnx.py](export_onnx.py) into `nanodet/tools` and run `export_onnx.py` to generate `nanodet-m.onnx`.
```
python tools/export_onnx.py
```

## 3.Build nanodet_trt Project
```
cd ../  # in project directory
mkdir build && cd build
cmake ..
make -j
```

## 4.run nanodet_trt
```
cd ../../bin/
./tensorrt_inference nanodet ../configs/nanodet/config.yaml ../samples/detection_segmentation
```

## 5.Results:
![](prediction.jpg)
