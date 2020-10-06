import argparse
import onnx
import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet

print('mxnet version:', mx.__version__)
print('onnx version:', onnx.__version__)
assert onnx.__version__ == '1.3.0'

parser = argparse.ArgumentParser(description='convert arcface models to onnx')
# general
parser.add_argument('--prefix', default='./model', help='prefix to load model.')
parser.add_argument('--epoch', default=0, type=int, help='epoch number to load model.')
parser.add_argument('--input_shape', nargs='+', default=[1, 3, 112, 112], type=int, help='input shape.')
parser.add_argument('--output_onnx', default='./arcface_r100.onnx', help='path to write onnx model.')
args = parser.parse_args()

input_shape = args.input_shape
print('input-shape:', input_shape)

sym, arg_params, aux_params = mx.model.load_checkpoint(args.prefix, args.epoch)

reshape_params = {}
for k, v in arg_params.items():
    if 'relu' in k:
        v = v.reshape(1, -1, 1, 1)
        reshape_params[k] = v
    else:
        reshape_params[k] = v

mx.model.save_checkpoint(args.prefix + "r", args.epoch + 1, sym, reshape_params, aux_params)

sym_file = f'{args.prefix + "r"}-symbol.json'
params_file = f'{args.prefix + "r"}-{args.epoch + 1:04d}.params'

converted_model_path = onnx_mxnet.export_model(sym_file, params_file, [input_shape], np.float32, args.output_onnx)
