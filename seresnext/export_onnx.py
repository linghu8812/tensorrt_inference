import onnx
import torch
from seresnext import se_resnext101

model = se_resnext101()
image = torch.zeros(1, 3, 224, 224)

torch.onnx.export(model, image, "se_resnext101_32x4d.onnx", input_names=['input'], output_names=['output'])

onnx_model = onnx.load("se_resnext101_32x4d.onnx")  # load onnx model
onnx.checker.check_model(onnx_model)
