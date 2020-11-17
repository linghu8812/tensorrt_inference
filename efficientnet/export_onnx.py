import onnx
import keras2onnx
import efficientnet.tfkeras as efn
import argparse

parser = argparse.ArgumentParser(description='Export efficientnet ONNX')
parser.add_argument('--batch_size', default=1, type=int, help='batch size.')
args = parser.parse_args()

model = efn.EfficientNetB0(weights='imagenet')

onnx_model = keras2onnx.convert_keras(model, model.name)
onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = args.batch_size
onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = args.batch_size
onnx.save_model(onnx_model, model.name + '.onnx')
