import onnx
import math
import argparse


parser = argparse.ArgumentParser(description='Export CenterFace ONNX')
parser.add_argument('--pretrained', help='pretrained centerface model', default='./centerface.onnx', type=str)
parser.add_argument('--input_shape', nargs='+', default=[1, 3, 640, 640], type=int, help='input shape.')
parser.add_argument('--onnx', help='onnx model', default='./centerface.onnx', type=str)
args = parser.parse_args()

model = onnx.load_model(args.pretrained)
input_shape = args.input_shape

d = model.graph.input[0].type.tensor_type.shape.dim
rate = (input_shape[2] / d[2].dim_value, input_shape[3] / d[3].dim_value)
print("rate: ", rate)
d[0].dim_value = input_shape[0]
d[2].dim_value = int(d[2].dim_value * rate[0])
d[3].dim_value = int(d[3].dim_value * rate[1])
for output in model.graph.output:
    d = output.type.tensor_type.shape.dim
    d[0].dim_value = input_shape[0]
    d[2].dim_value = int(d[2].dim_value * rate[0])
    d[3].dim_value = int(d[3].dim_value * rate[1])

onnx.save_model(model, args.onnx)
