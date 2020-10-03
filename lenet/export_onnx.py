import onnx
import torch

# export from pytorch to onnx
net = torch.load('mnist_net.pt').to('cpu')
image = torch.randn(10, 1, 28, 28)
torch.onnx.export(net, image, 'mnist_net.onnx', input_names=['input'], output_names=['output'])

# check onnx model
onnx_model = onnx.load("mnist_net.onnx")  # load onnx model
onnx.checker.check_model(onnx_model)
