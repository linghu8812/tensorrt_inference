import onnx
import onnxsim
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Export CenterFace ONNX')
parser.add_argument('--pretrained', help='pretrained centerface model', default='./centerface.onnx', type=str)
parser.add_argument('--input_shape', nargs='+', default=[1, 3, 640, 640], type=int, help='input shape.')
parser.add_argument('--onnx', help='onnx model', default='./centerface_transpose.onnx', type=str)
args = parser.parse_args()

model = onnx.load_model(args.pretrained)
input_shape = args.input_shape

current_output = []
d = model.graph.input[0].type.tensor_type.shape.dim
rate = (input_shape[2] / d[2].dim_value, input_shape[3] / d[3].dim_value)
print("rate: ", rate)
d[0].dim_value = input_shape[0]
d[2].dim_value = int(d[2].dim_value * rate[0])
d[3].dim_value = int(d[3].dim_value * rate[1])

dim1_value = 0
for output in model.graph.output:
    d = output.type.tensor_type.shape.dim
    d[0].dim_value = input_shape[0]
    d[2].dim_value = int(d[2].dim_value * rate[0])
    d[3].dim_value = int(d[3].dim_value * rate[1])
    dim1_value += d[1].dim_value
    current_output.append(output.name)

concat_name = 'concat'
concat_node = onnx.helper.make_node(
    'Concat',
    axis=1,
    inputs=current_output,
    outputs=[concat_name],
    name=concat_name,
)

model.graph.node.append(concat_node)

reshape_name = 'reshape'
reshape_node = onnx.helper.make_node(
    'Reshape',
    inputs=[concat_name, 'reshape_params'],
    outputs=[reshape_name]
)
reshape_param = np.array([input_shape[0], dim1_value, -1]).astype(np.int64)
scale_init = onnx.helper.make_tensor('reshape_params', onnx.TensorProto.INT64, reshape_param.shape, reshape_param)
scale_input = onnx.helper.make_tensor_value_info('reshape_params', onnx.TensorProto.INT64, reshape_param.shape)
model.graph.initializer.append(scale_init)
model.graph.input.append(scale_input)
model.graph.node.append(reshape_node)

transpose_name = 'transpose'
transpose_node = onnx.helper.make_node(
    'Transpose',
    inputs=[reshape_name],
    outputs=[transpose_name],
    perm=(0, 2, 1)
)
model.graph.node.append(transpose_node)
#
for index in range(len(model.graph.output) - 1, -1, -1):
    if index == 0:
        model.graph.output[index].name = transpose_name
    else:
        model.graph.output.remove(model.graph.output[index])

for output in model.graph.output:
    d = output.type.tensor_type.shape.dim
    d[1].dim_value = d[2].dim_value * d[3].dim_value
    d[2].dim_value = dim1_value
    d.remove(d[3])

onnx.save_model(model, args.onnx)
model_onnx = onnx.load(args.onnx)  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model
model_onnx, check = onnxsim.simplify(model_onnx)
assert check, 'assert check failed'
onnx.save(model_onnx, args.onnx)
