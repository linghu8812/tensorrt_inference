import onnx
from gluoncv import model_zoo
from mxnet import nd
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import numpy as np
import mxnet.contrib.onnx.mx2onnx.export_onnx as mx_op
import argparse


def get_sym_train(symbol):
    model_group = []
    for layers in symbol:
        bilinear_transpose = mx.symbol.transpose(layers, (0, 2, 3, 1))
        model_group.append(bilinear_transpose)
    # return mx.sym.Group(model_group)
    add_layer = mx.symbol.add_n(model_group[0], model_group[1])
    return add_layer


def get_scales(network, infer_size):
    all_layers = network.get_internals()
    _, out_shape, _ = all_layers.infer_shape(data=infer_size)
    outputs = all_layers.list_outputs()
    scale_dict = {}
    for index, layer in enumerate(outputs):
        if layer.endswith('output') and (network[0].name in layer or network[1].name in layer):
            pre_shape = out_shape[index - 2]
            cur_shape = out_shape[index - 1]
            scale_dict[outputs[index - 1][:-7]] = np.array(cur_shape) / np.array(pre_shape)
    return scale_dict


def get_inputs(node, kwargs):
    """Helper function to get inputs"""
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    index_lookup = kwargs["index_lookup"]
    inputs = node["inputs"]
    attrs = node.get("attrs", {})

    input_nodes = []
    for ip in inputs:
        input_node_id = index_lookup[ip[0]]
        input_nodes.append(proc_nodes[input_node_id].name)

    return name, input_nodes, attrs


def create_helper_tensor_node(input_vals, output_name, kwargs):
    """create extra tensor node from numpy values"""
    data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input_vals.dtype]
    tensor_node = onnx.helper.make_tensor_value_info(
        name=output_name,
        elem_type=data_type,
        shape=input_vals.shape
    )
    kwargs["initializer"].append(
        onnx.helper.make_tensor(
            name=output_name,
            data_type=data_type,
            dims=input_vals.shape,
            vals=input_vals.flatten().tolist(),
            raw=False,
        )
    )
    return tensor_node


@mx_op.MXNetGraph.register("_contrib_BilinearResize2D")
def convert_bilinearresize2d(node, **kwargs):
    """
    Map MXNet's UpSampling operator attributes to onnx's Upsample operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    sample_type = attrs.get('sample_type', 'linear')
    height = float(6)
    width = float(6)
    scales = np.array([1.0, 1.0, height, width], dtype=np.float32)
    roi = np.array([], dtype=np.float32)
    node_roi = create_helper_tensor_node(roi, name + 'roi', kwargs)
    node_scales = create_helper_tensor_node(scales, name + 'scales', kwargs)
    node = onnx.helper.make_node(
        'Resize',
        inputs=[input_nodes[0], name + 'roi', name + 'scales'],
        outputs=[name],
        coordinate_transformation_mode='asymmetric',
        mode=sample_type,
        nearest_mode='floor',
        name=name
    )
    return [node_roi, node_scales, node]


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


def main():
    parser = argparse.ArgumentParser(description='convert arcface models to onnx')
    # general
    parser.add_argument('--prefix', default='fcn_resnet101_voc', help='prefix to load model.')
    parser.add_argument('--epoch', default=0, type=int, help='epoch number to load model.')
    parser.add_argument('--input_shape', nargs='+', default=[1, 3, 640, 640], type=int, help='input shape.')
    args = parser.parse_args()
    converted_onnx_filename = args.prefix + '.onnx'
    net = model_zoo.get_model(args.prefix, pretrained=True)
    x = nd.zeros(args.input_shape)
    net.hybridize()
    net(x)
    net.export(args.prefix)
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.prefix, args.epoch)
    model = get_sym_train(sym)
    # scale_dict = get_scales(model, args.input_shape)
    mx.model.save_checkpoint(args.prefix + '_transpose', args.epoch, model, arg_params, aux_params)
    converted_onnx_filename = onnx_mxnet.export_model(args.prefix + '_transpose-symbol.json',
                                                      f'{args.prefix}_transpose-{args.epoch:04d}.params',
                                                      [args.input_shape], np.float32, converted_onnx_filename)

    model_proto = onnx.load(converted_onnx_filename)
    onnx.checker.check_model(model_proto)


if __name__ == '__main__':
    main()
