import torch

net = torch.load('mnist_net.pt').to('cpu')

image = torch.randn(1, 1, 28, 28)

torch.onnx.export(net, image, 'mnist_net.onnx', input_names=['input'], output_names=['output'])
