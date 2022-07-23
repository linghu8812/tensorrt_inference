import argparse
import onnx
import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
import mxnet.contrib.onnx.mx2onnx.export_onnx as mx_op
from mxnet.contrib.onnx.mx2onnx._op_translations import get_inputs


@mx_op.MXNetGraph.register("BatchNorm")
def convert_batchnorm(node, **kwargs):
    """
    Map MXNet's BatchNorm operator attributes to onnx's BatchNormalization operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    momentum = float(attrs.get("momentum", 0.9))
    eps = float(attrs.get("eps", 0.001))
    bn_node = onnx.helper.make_node(
        "BatchNormalization",
        input_nodes,
        [name],
        name=name,
        epsilon=eps,
        momentum=momentum,
        # MXNet computes mean and variance per feature for batchnorm
        # Default for onnx is across all spatial features. So disabling the parameter.
        # spatial=0
    )
    return [bn_node]


@mx_op.MXNetGraph.register("LeakyReLU")
def convert_leakyrelu(node, **kwargs):
    """Map MXNet's LeakyReLU operator attributes to onnx's Elu/LeakyRelu/PRelu operators
    based on the input node's attributes and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    initializer = kwargs["initializer"]
    act_type = attrs.get("act_type", "leaky")
    alpha = float(attrs.get("slope", 0.25))
    act_name = {"elu": "Elu", "leaky": "LeakyRelu", "prelu": "PRelu",
                "selu": "Selu"}
    reshape_val_name = 'reshape' + str(kwargs["idx"])
    input_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')]
    reshape_value = np.array([1, -1, 1, 1], dtype='int64')
    dims = np.shape(reshape_value)
    shape_node = onnx.helper.make_tensor_value_info(reshape_val_name, input_type, dims)
    initializer.append(
        onnx.helper.make_tensor(
            name=reshape_val_name,
            data_type=input_type,
            dims=dims,
            vals=reshape_value,
            raw=False,
        )
    )
    slope_op_name = 'slope' + str(kwargs["idx"])
    lr_node = []
    if act_type == "prelu" or act_type == "selu":
        reshape_slope_node = onnx.helper.make_node(
            'Reshape',
            inputs=[input_nodes[1], reshape_val_name],
            outputs=[slope_op_name],
            name=slope_op_name
        )
        node = onnx.helper.make_node(
            act_name[act_type],
            inputs=[input_nodes[0], slope_op_name],
            outputs=[name],
            name=name)
        lr_node.append(shape_node)
        lr_node.append(reshape_slope_node)
        lr_node.append(node)
    else:
        node = onnx.helper.make_node(
            act_name[act_type],
            inputs=input_nodes,
            outputs=[name],
            name=name,
            alpha=alpha)
        lr_node.append(node)
    return lr_node


print('mxnet version:', mx.__version__)
print('onnx version:', onnx.__version__)

parser = argparse.ArgumentParser(description='convert arcface models to onnx')
# general
parser.add_argument('--prefix', default='./model/gender-age', help='prefix to load model.')
parser.add_argument('--epoch', default=0, type=int, help='epoch number to load model.')
parser.add_argument('--input_shape', nargs='+', default=[1, 3, 112, 112], type=int, help='input shape.')
parser.add_argument('--output_onnx', default='./gender-age.onnx', help='path to write onnx model.')
args = parser.parse_args()

input_shape = args.input_shape
print('input-shape:', input_shape)

sym, arg_params, aux_params = mx.model.load_checkpoint(args.prefix, args.epoch)
all_layers = sym.get_internals()
for layer in all_layers:
    if 'gamma' in layer.name and layer.attr('fix_gamma') == 'True':
        arg_params[layer.name] = mx.nd.array(np.ones(arg_params[layer.name].shape))

mx.model.save_checkpoint(args.prefix + "r", args.epoch + 1, sym, arg_params, aux_params)

sym_file = f'{args.prefix + "r"}-symbol.json'
params_file = f'{args.prefix + "r"}-{args.epoch + 1:04d}.params'

converted_model_path = onnx_mxnet.export_model(sym_file, params_file, [input_shape], np.float32, args.output_onnx)
