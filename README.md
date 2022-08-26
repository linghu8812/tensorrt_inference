# **TensorRT Models Deploy from ONNX**

[![OSCS Status](https://www.oscs1024.com/platform/badge/linghu8812/tensorrt_inference.svg?size=small)](https://www.oscs1024.com/project/linghu8812/tensorrt_inference?ref=badge_small)

## **Install Depends**
see [INSTALL.md](INSTALL.md)

## **Build from Docker**
```
docker build -t tensorrt_inference:0.1.0_rc .
```

## **Supported Models**

download transfered ONNX models: [https://pan.baidu.com/s/1Ff_SA9Q66DUnZjSipPa74Q](https://pan.baidu.com/s/1Ff_SA9Q66DUnZjSipPa74Q), code: `opkp`
![](samples/resources/image.png)

models|framework|instruction
---|---|---
[lenet](project/lenet)|PyTorch|An example from model training to TensorRT model deploy
[alexnet](project/alexnet)|MXNet Gluon|MXNet Gluon example
[arcface](project/arcface)|MXNet Symbol|MXNet Symbol and face recognition example
[CenterFace](project/CenterFace)|ONNX|rewrite ONNX model and face detection example
[efficientnet](project/efficientnet)|Keras|Keras to ONNX example
[face_alignment](project/face_alignment)|MXNet Symbol|MXNet Symbol and face key points  detection example
[fast-reid](project/fast-reid)|PyTorch|PyTorch and pedestrian reid example
[FCN](project/FCN)|GluonCV|MXNet GluonCV semantic segmentation example
[gender-age](project/gender-age)|MXNet Symbol|MXNet Symbol and face gender and age recognize example
[ghostnet](project/ghostnet)|PyTorch|PyTorch example
[MiniFASNet](project/MiniFASNet)|PyTorch|PyTorch face anti spoofing example
[mmpose](project/mmpose)|PyTorch|PyTorch person key points detect example
[nanodet](project/nanodet)|PyTorch|PyTorchlightweight anchor-free object detection example 
[RetinaFace](project/RetinaFace)|MXNet Symbol|MXNet Symbol and face detection example
[ScaledYOLOv4](project/ScaledYOLOv4)|PyTorch|YOLOv4 large with PyTorch implementation
[scrfd](project/scrfd)|PyTorch|PyTorch scrfd face detection example
[seresnext](project/seresnext)|PyTorch|PyTorch example
[Swin-Transformer](project/Swin-Transformer)|timm|timm image classification example
[yolor](project/yolor)|PyTorch|PyTorch and object detection example
[Yolov4](project/Yolov4)|darknet|darknet and object detection example
[yolov5](project/yolov5)|PyTorch|PyTorch and object detection example
[YOLOv6](project/YOLOv6)|PyTorch|PyTorch and object detection example
[yolov7](project/yolov7)|PyTorch|PyTorch and object detection example
 
 ## **Discussion Groups**

 - **QQ group:** 588966055

 ![](samples/resources/group.jpg)
