import onnx
import torch
import timm

model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
image = torch.zeros(1, 3, 224, 224)

torch.onnx.export(model, image, "swin_tiny_patch4_window7_224.onnx", input_names=['input'], output_names=['output'],
                  opset_version=12)

onnx_model = onnx.load("swin_tiny_patch4_window7_224.onnx")  # load onnx model
onnx.checker.check_model(onnx_model)
