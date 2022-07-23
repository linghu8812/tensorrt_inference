import onnx
import torch
from ghostnet import ghostnet

model = ghostnet()
model.load_state_dict(torch.load('./models/state_dict_93.98.pth'))
image = torch.zeros(1, 3, 224, 224)

torch.onnx.export(model, image, "ghostnet.onnx", input_names=['input'], output_names=['output'])

onnx_model = onnx.load("ghostnet.onnx")  # load onnx model
onnx.checker.check_model(onnx_model)
