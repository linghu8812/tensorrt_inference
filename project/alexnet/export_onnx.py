import onnx
from mxnet.gluon.model_zoo import vision as models
from mxnet import nd

from mxnet.contrib import onnx as onnx_mxnet
import numpy as np

net = models.alexnet(pretrained=True)
converted_onnx_filename = 'alexnet.onnx'

# Export MXNet model to ONNX format via MXNet's export_model API
x = nd.zeros((1, 3, 224, 224))
net.hybridize()
net(x)
net.export('alexnet')
converted_onnx_filename = onnx_mxnet.export_model('alexnet-symbol.json', 'alexnet-0000.params',
                                                  [(1, 3, 224, 224)], np.float32, converted_onnx_filename)

model_proto = onnx.load(converted_onnx_filename)
onnx.checker.check_model(model_proto)
