import onnx
import mxnet as mx
from retinaface import RetinaFace
# from retinaface_cov import RetinaFaceCoV
from mxnet.contrib import onnx as onnx_mxnet
import mxnet.contrib.onnx.mx2onnx.export_onnx as mx_op
from mxnet.contrib.onnx.mx2onnx._op_translations import get_inputs, convert_string_to_list
import numpy as np
import argparse
import json


def get_sym_train(symbol):
    score_group = []
    box_group = []
    landmark_group = []
    mask_group = []
    for layers in symbol:
        transpose_layers = mx.symbol.transpose(layers, (0, 2, 3, 1))
        if 'face_rpn_cls_prob_reshape_stride' in layers.name:
            slice_layers = mx.symbol.slice_axis(transpose_layers, axis=3, begin=2, end=4)
            reshape_layers = mx.symbol.Reshape(data=slice_layers,
                                               shape=(0, -1, 1),
                                               name=layers.name + 'transpose')
            score_group.append(reshape_layers)
        if 'face_rpn_bbox_pred_stride' in layers.name:
            reshape_layers = mx.symbol.Reshape(data=transpose_layers,
                                               shape=(0, -1, 4),
                                               name=layers.name + 'transpose')
            box_group.append(reshape_layers)
        if 'face_rpn_landmark_pred_stride' in layers.name:
            reshape_layers = mx.symbol.Reshape(data=transpose_layers,
                                               shape=(0, -1, 10),
                                               name=layers.name + 'transpose')
            landmark_group.append(reshape_layers)
        if 'face_rpn_type_prob_reshape_stride' in layers.name:
            slice_layers = mx.symbol.slice_axis(transpose_layers, axis=3, begin=4, end=6)
            reshape_layers = mx.symbol.Reshape(data=slice_layers,
                                               shape=(0, -1, 1),
                                               name=layers.name + 'transpose')
            mask_group.append(reshape_layers)
    score_concat = mx.sym.concat(*score_group, dim=1, name='score_concat')
    bbox_concat = mx.sym.concat(*box_group, dim=1, name='bbox_concat')
    landmark_concat = mx.sym.concat(*landmark_group, dim=1, name='landmark_concat')
    if len(mask_group) > 0:
        mask_concat = mx.sym.concat(*mask_group, dim=1, name='mask_concat')
        output = mx.sym.concat(*[score_concat, bbox_concat, landmark_concat, mask_concat], dim=2, name='output')
    else:
        output = mx.sym.concat(*[score_concat, bbox_concat, landmark_concat], dim=2, name='output')
    return output


def change_plus(file_prefix):
    file_name = file_prefix + '-symbol.json'
    with open(file_name, 'r') as f:
        model_dict = json.load(f)
    index = 0
    for node in model_dict['nodes']:
        if 'plus' in node['name']:
            node['name'] = '_plus' + str(index)
            index += 1
    with open(file_name, 'w') as f:
        json.dump(model_dict, f, indent=1)


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


def create_helper_shape_node(input_node, node_name):
    """create extra transpose node for dot operator"""
    trans_node = onnx.helper.make_node(
        'Shape',
        inputs=[input_node],
        outputs=[node_name],
        name=node_name
    )
    return trans_node


@mx_op.MXNetGraph.register("SoftmaxActivation")
def convert_softmax_activation(node, **kwargs):
    """
    Map MXNet's softmax operator attributes to onnx's Softmax operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    outputs = []
    mode = attrs.get("mode", "channel")
    axis = -1
    if mode == "channel":
        # transpose nchw -> nhwc  Softmax: axis=3  nhwc -> nchw
        trans_op_name = 'transpose' + str(kwargs["idx"])
        trans_node = onnx.helper.make_node(
            "Transpose",
            input_nodes,
            [trans_op_name],
            name=trans_op_name,
            perm=[0, 2, 3, 1]
        )
        softmax_op_name = 'softmax' + str(kwargs["idx"])
        softmax_node = onnx.helper.make_node(
            "Softmax",
            [trans_op_name],
            [softmax_op_name],
            axis=3,
            name=softmax_op_name
        )
        output_node = onnx.helper.make_node(
            "Transpose",
            [softmax_op_name],
            [name],
            name=name,
            perm=[0, 3, 1, 2]
        )
        outputs.append(trans_node)
        outputs.append(softmax_node)
        outputs.append(output_node)
    else:
        softmax_node = onnx.helper.make_node(
            "Softmax",
            input_nodes,
            [name],
            axis=axis,
            name=name
        )
        outputs.append(softmax_node)
    return outputs


@mx_op.MXNetGraph.register("UpSampling")
def convert_upsample(node, **kwargs):
    """
    Map MXNet's UpSampling operator attributes to onnx's Upsample operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    sample_type = attrs.get('sample_type', 'nearest')
    sample_type = 'linear' if sample_type == 'bilinear' else sample_type
    scale = convert_string_to_list(attrs.get('scale'))
    scaleh = scalew = float(scale[0])
    if len(scale) > 1:
        scaleh = float(scale[0])
        scalew = float(scale[1])
    scale = np.array([1.0, 1.0, scaleh, scalew], dtype=np.float32)
    scale_node = create_helper_tensor_node(scale, name + 'scales', kwargs)
    input_nodes.append(name + 'scales')
    node = onnx.helper.make_node(
        'Resize',
        input_nodes,
        [name],
        mode=sample_type,
        name=name
    )
    return [scale_node, node]


@mx_op.MXNetGraph.register("Crop")
def convert_crop(node, **kwargs):
    """Map MXNet's crop operator attributes to onnx's Crop operator
    and return the created node.
    """
    name, inputs, attrs = get_inputs(node, kwargs)
    start = np.array([0, 0, 0, 0], dtype=np.int)  # index是int类型
    start_node = create_helper_tensor_node(start, name + '_starts', kwargs)
    shape_node = create_helper_shape_node(inputs[1], inputs[1] + '_shape')
    crop_node = onnx.helper.make_node(
        "Slice",
        inputs=[inputs[0], name + '_starts', inputs[1] + '_shape'],  # data、start、end
        outputs=[name],
        name=name
    )
    return [start_node, shape_node, crop_node]


@mx_op.MXNetGraph.register("BatchNorm")
def convert_batchnorm(node, **kwargs):
    """Map MXNet's BatchNorm operator attributes to onnx's BatchNormalization operator
    and return the created node.
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
        momentum=momentum
        # MXNet computes mean and variance per channel for batchnorm.
        # Default for onnx is across all spatial features. Relying on default
        # ONNX behavior of spatial=1 for ONNX opset 8 and below. As the spatial
        # attribute is deprecated in opset 9 and above, not explicitly encoding it.
    )
    return [bn_node]


@mx_op.MXNetGraph.register("slice_axis")
def convert_slice_axis(node, **kwargs):
    """Map MXNet's slice_axis operator attributes to onnx's Slice operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axes = int(attrs.get("axis"))
    starts = int(attrs.get("begin"))
    ends = attrs.get("end", None)
    if not ends:
        raise ValueError("Slice: ONNX doesnt't support 'None' in 'end' attribute")

    export_nodes = []

    starts = np.atleast_1d(np.asarray(starts, dtype=np.int))
    ends = np.atleast_1d(np.asarray(ends, dtype=np.int))
    axes = np.atleast_1d(np.asarray(axes, dtype=np.int))

    starts_node = create_helper_tensor_node(starts, name + '__starts', kwargs)
    export_nodes.append(starts_node)
    starts_node = starts_node.name

    ends_node = create_helper_tensor_node(ends, name + '__ends', kwargs)
    export_nodes.append(ends_node)
    ends_node = ends_node.name

    axes_node = create_helper_tensor_node(axes, name + '__axes', kwargs)
    export_nodes.append(axes_node)
    axes_node = axes_node.name

    input_node = input_nodes[0]
    node = onnx.helper.make_node(
        "Slice",
        [input_node, starts_node, ends_node, axes_node],
        [name],
        name=name,
    )
    export_nodes.extend([node])

    return export_nodes


def main():
    parser = argparse.ArgumentParser(description='convert arcface models to onnx')
    # general
    parser.add_argument('--prefix', default='./model/R50', help='prefix to load model.')
    parser.add_argument('--epoch', default=0, type=int, help='epoch number to load model.')
    parser.add_argument('--gpuid', default=0, type=int, help='ctx_id in model.')
    parser.add_argument('--network', default='net3', type=str, help='network in model.')
    parser.add_argument('--input_shape', nargs='+', default=[1, 3, 640, 640], type=int, help='input shape.')
    args = parser.parse_args()
    converted_onnx_filename = args.prefix + '.onnx'
    model = RetinaFace(args.prefix, args.epoch, args.gpuid, args.network).model
    # model = RetinaFaceCoV(args.prefix, args.epoch, args.gpuid, args.network).model
    sym, arg_params, aux_params = model.symbol, model._arg_params, model._aux_params
    model = get_sym_train(sym)
    mx.model.save_checkpoint(args.prefix + '_transpose', args.epoch, model, arg_params, aux_params)
    change_plus(args.prefix + '_transpose')
    converted_onnx_filename = onnx_mxnet.export_model(args.prefix + '_transpose-symbol.json',
                                                      f'{args.prefix}_transpose-{args.epoch:04d}.params',
                                                      [args.input_shape], np.float32, converted_onnx_filename)


if __name__ == '__main__':
    main()
