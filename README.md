# **TensorRT Models Deploy from ONNX**

## **Install Depends**
see [INSTALL.md](INSTALL.md)

## **Build from Docker**
```
docker build -t tensorrt_inference:0.1.0_rc .
```

## **Supported Models**

models|framework|instruction
---|---|---
[lenet](lenet)|PyTorch|An example from model training to TensorRT model deploy
[alexnet](alexnet)|MXNet Gluon|MXNet Gluon example
[arcface](arcface)|MXNet Symbol|MXNet Symbol and face recognition example
[CenterFace](CenterFace)|ONNX|rewrite ONNX model and face detection example
[efficientnet](efficientnet)|Keras|Keras to ONNX example
[face_alignment](face_alignment)|MXNet Symbol|MXNet Symbol and face key points  detection example
[FCN](FCN)|GluonCV|MXNet GluonCV semantic segmentation example
[gender-age](gender-age)|MXNet Symbol|MXNet Symbol and face gender and age recognize example
[ghostnet](ghostnet)|PyTorch|PyTorch example
[MiniFASNet](MiniFASNet)|PyTorch|PyTorch face anti spoofing example
[nanodet](nanodet)|PyTorch|PyTorchlightweight anchor-free object detection example 
[RetinaFace](RetinaFace)|MXNet Symbol|MXNet Symbol and face detection example
[ScaledYOLOv4](ScaledYOLOv4)|PyTorch|YOLOv4 large with PyTorch implementation
[seresnext](seresnext)|PyTorch|PyTorch example
[Yolov4](Yolov4)|darknet|darknet and object detection example
[yolov5](yolov5)|PyTorch|PyTorch and object detection example
 